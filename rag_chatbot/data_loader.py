from typing import List, Tuple
import pandas as pd
import numpy as np
import re
from pathlib import Path
from . import config
try:
    from langchain.text_splitter import TokenTextSplitter
except Exception:
    # Provide a lightweight fallback TokenTextSplitter to avoid hard dependency on langchain
    class TokenTextSplitter:
        def __init__(self, chunk_size: int = 2048, chunk_overlap: int = 128):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str) -> List[str]:
            if not text:
                return []
            # Naive character-based splitter with overlap
            chunks = []
            i = 0
            L = len(text)
            while i < L:
                end = min(i + self.chunk_size, L)
                chunks.append(text[i:end])
                if end == L:
                    break
                i = max(0, end - self.chunk_overlap)
            return chunks
from pandas.errors import ParserError
import io


def _clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    # Normalize whitespace and remove excessive newlines
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_and_preprocess(path: Path = None) -> pd.DataFrame:
    """Load CSV and preprocess types and text normalization.

    This function will try common delimiters (comma, semicolon) and fall back to
    automatic detection with the python engine if necessary.
    """
    p = Path(path or config.DATA_PATH)

    def _read_single_csv(file_path: Path) -> pd.DataFrame:
        """Robustly read a single CSV file trying common encodings and separators."""
        tried = []
        df_local = None
        encodings = ["utf-8", "latin-1"]
        for enc in encodings:
            try:
                tried.append((",", enc))
                df_local = pd.read_csv(file_path, sep=",", encoding=enc, low_memory=False)
                break
            except ParserError:
                try:
                    tried.append((";", enc))
                    df_local = pd.read_csv(file_path, sep=";", encoding=enc, low_memory=False)
                    break
                except ParserError:
                    try:
                        tried.append(("auto", enc))
                        with open(file_path, "r", encoding=enc, errors="replace") as f:
                            sample = f.read()
                        df_local = pd.read_csv(io.StringIO(sample), sep=None, engine="python")
                        break
                    except Exception:
                        df_local = None
        if df_local is None:
            # Final attempt: let pandas try with default encoding and python engine
            df_local = pd.read_csv(file_path, sep=None, engine="python", encoding="utf-8", errors="replace")
        return df_local

    # If the configured path is a directory, load and concatenate all CSV files inside
    if p.is_dir():
        csv_files = sorted([f for f in p.glob("**/*.csv") if f.is_file()])
        if not csv_files:
            # Fallback: try project-level fallback data files (chatbotdf.csv or BASE.xlsx)
            repo_root = Path(__file__).resolve().parent.parent
            fallback_csv = repo_root / "chatbotdf.csv"
            fallback_xlsx = repo_root / "BASE.xlsx"
            if fallback_csv.exists():
                try:
                    df = _read_single_csv(fallback_csv)
                except Exception:
                    df = None
            elif fallback_xlsx.exists():
                try:
                    df = pd.read_excel(fallback_xlsx)
                except Exception:
                    df = None
            else:
                raise FileNotFoundError(f"No CSV files found in directory: {p}")
            if df is None or df.empty:
                raise RuntimeError(f"Failed to read any CSV files from {p} or fallback files")
        else:
            dfs = []
            for f in csv_files:
                try:
                    dfs.append(_read_single_csv(f))
                except Exception:
                    # skip problematic files but continue processing others
                    continue
            if not dfs:
                raise RuntimeError(f"Failed to read any CSV files from {p}")
            df = pd.concat(dfs, ignore_index=True, sort=False)
    else:
        # single file path
        df = _read_single_csv(p)


    # Standardize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Clean text columns
    text_cols = [
        "Client_Principal",
        "Intitule_client",
        "Categorie_Client",
        "Pays",
        "Zone",
        "Gouvernorat",
        "Adresse",
        "Representant",
        "Marque",
        "Famille",
        "Sous_Famille",
        "Designation",
    ]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].apply(_clean_text)

    # Dates
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        except Exception:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Numeric conversions
    numeric_cols = [
        "Qte_Vendu",
        "CA_HT_BRUT",
        "Tx_Remise",
        "CA_HT_NET",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Year column fallback
    if "Annee" in df.columns and df["Annee"].dtype == object:
        df["Annee"] = pd.to_numeric(df["Annee"], errors="coerce")

    return df


def df_to_documents(df: pd.DataFrame, chunk_size: int = config.CHUNK_SIZE) -> List[dict]:
    """Convert rows into document dicts for embedding. Combine relevant fields."""
    docs = []
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=config.CHUNK_OVERLAP)
    # Build a lowercase column map so lookups are case-insensitive and tolerant to variants
    col_map = {str(c).lower(): c for c in df.columns}

    def _lookup(row, *names):
        """Return the first matching value from row for any of the provided column name variants (case-insensitive)."""
        for n in names:
            if n is None:
                continue
            # direct match
            if n in df.columns:
                v = row.get(n)
                if v is not None:
                    return v
            # case-insensitive
            key = str(n).lower()
            if key in col_map:
                v = row.get(col_map[key])
                if v is not None:
                    return v
        return None

    for idx, row in df.iterrows():
        parts = []

        # Core identifiers (case-insensitive): client fields
        for c in ("Client_Principal", "Intitule_client", "Code_Client", "client_principal", "client"):
            v = _lookup(row, c)
            if v:
                parts.append(f"{c}: {v}")

        # Product and category - accept common lowercased variants returned by forecasts
        for c in ("Marque", "marque", "Famille", "famille", "Sous_Famille", "Ref_Article", "ref_article", "Designation", "designation"):
            v = _lookup(row, c)
            if v:
                parts.append(f"{c}: {v}")

        # Sales metrics
        for c in ("Qte_Vendu", "qte_vendu", "CA_HT_BRUT", "CA_HT_NET", "ca_ht_net", "Tx_Remise"):
            v = _lookup(row, c)
            if v is not None:
                parts.append(f"{c}: {v}")

        # Date info
        date_v = _lookup(row, "Date", "date")
        if date_v is not None and not pd.isna(date_v):
            try:
                parts.append(f"Date: {pd.to_datetime(date_v).date()}")
            except Exception:
                parts.append(f"Date: {date_v}")
        else:
            mois = _lookup(row, "Mois", "mois")
            annee = _lookup(row, "Annee", "annee", "next_year")
            if mois is not None and annee is not None:
                parts.append(f"Mois: {mois} Annee: {annee}")

        # Forecasts (if present) - include avg and per-method forecasts so RAG can reason about predictions
        forecast_cols = [
            'avg_forecast', 'sma_forecast', 'es_forecast', 'lr_forecast',
            'arima_forecast', 'prophet_forecast', 'xgb_forecast', 'next_year'
        ]
        for c in forecast_cols:
            v = _lookup(row, c)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                parts.append(f"{c}: {v}")

        content = " | ".join(parts)
        if not content:
            continue
        # Use token splitter to generate well-sized chunks
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            # assemble metadata using tolerant lookups
            year_val = _lookup(row, "Annee", "annee", "next_year")
            try:
                year_int = int(year_val) if year_val is not None and str(year_val).isdigit() else None
            except Exception:
                year_int = None

            docs.append({
                "content": chunk,
                "metadata": {
                    "source_row": int(idx),
                    "client": _lookup(row, "Client_Principal", "client", "Intitule_client"),
                    "year": year_int,
                    "chunk_index": i,
                    # include forecast metadata when available
                    "avg_forecast": _lookup(row, "avg_forecast"),
                    "sma_forecast": _lookup(row, "sma_forecast"),
                    "arima_forecast": _lookup(row, "arima_forecast"),
                    "prophet_forecast": _lookup(row, "prophet_forecast"),
                    "xgb_forecast": _lookup(row, "xgb_forecast"),
                },
            })

    return docs
