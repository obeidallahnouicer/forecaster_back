from typing import List, Tuple
import pandas as pd
import numpy as np
import re
from pathlib import Path
from . import config
from langchain.text_splitter import TokenTextSplitter
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
    p = path or config.DATA_PATH
    # Try a few common separators to be robust against different CSV exports
    tried = []
    df = None
    encodings = ["utf-8", "latin-1"]
    for enc in encodings:
        try:
            tried.append((",", enc))
            df = pd.read_csv(p, sep=",", encoding=enc, low_memory=False)
            used_sep = ","
            break
        except ParserError:
            try:
                tried.append((";", enc))
                df = pd.read_csv(p, sep=";", encoding=enc, low_memory=False)
                used_sep = ";"
                break
            except ParserError:
                # Try python engine autodetect
                try:
                    tried.append(("auto", enc))
                    with open(p, "r", encoding=enc, errors="replace") as f:
                        sample = f.read()
                    df = pd.read_csv(io.StringIO(sample), sep=None, engine="python")
                    used_sep = "auto"
                    break
                except Exception:
                    df = None
    if df is None:
        # Final attempt: let pandas try with default encoding and python engine
        df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8", errors="replace")


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

    for idx, row in df.iterrows():
        parts = []
        # Core identifiers
        for c in ["Client_Principal", "Intitule_client", "Code_Client"]:
            if c in df.columns and row.get(c):
                parts.append(f"{c}: {row.get(c)}")

        # Product and category
        for c in ["Marque", "Famille", "Sous_Famille", "Ref_Article", "Designation"]:
            if c in df.columns and row.get(c):
                parts.append(f"{c}: {row.get(c)}")

        # Sales metrics
        for c in ["Qte_Vendu", "CA_HT_BRUT", "Tx_Remise", "CA_HT_NET"]:
            if c in df.columns:
                parts.append(f"{c}: {row.get(c)}")

        # Date info
        if "Date" in df.columns and not pd.isna(row.get("Date")):
            parts.append(f"Date: {row.get('Date').date()}")
        else:
            if "Mois" in df.columns and "Annee" in df.columns:
                parts.append(f"Mois: {row.get('Mois')} Annee: {row.get('Annee')}")

        content = " | ".join(parts)
        if not content:
            continue
        # Use token splitter to generate well-sized chunks
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            docs.append({
                "content": chunk,
                "metadata": {
                    "source_row": int(idx),
                    "client": row.get("Client_Principal"),
                    "year": int(row.get("Annee")) if row.get("Annee") is not None and str(row.get("Annee")).isdigit() else None,
                    "chunk_index": i,
                },
            })

    return docs
