from pathlib import Path
import uuid
import shutil
import pandas as pd
from typing import Dict, Optional
from sales_forecaster import SalesForecaster
import hashlib
import os

WORKDIR = Path("./server_data")
WORKDIR.mkdir(parents=True, exist_ok=True)


class SessionInfo:
    def __init__(self, session_id: str, file_path: str, forecaster: SalesForecaster):
        self.session_id = session_id
        self.file_path = file_path
        self.forecaster = forecaster


class Registry:
    """Manage sessions and their SalesForecaster instances."""

    def __init__(self):
        self._sessions: Dict[str, SessionInfo] = {}

    def create_session_from_file(self, file_path: str, frequency: str = "yearly") -> SessionInfo:
        sid = str(uuid.uuid4())
        out_dir = WORKDIR / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / Path(file_path).name
        shutil.copy(file_path, dest)

        # Validate frequency
        if frequency.lower() not in ["yearly", "monthly"]:
            raise RuntimeError(f"Invalid frequency '{frequency}'. Must be 'yearly' or 'monthly'")

        # load with encoding fallbacks for CSVs
        try:
            if dest.suffix.lower() == ".csv":
                # try common encodings
                try:
                    df = pd.read_csv(dest, encoding="utf-8")
                except Exception:
                    try:
                        df = pd.read_csv(dest, encoding="latin-1")
                    except Exception:
                        df = pd.read_csv(dest, encoding="cp1252")
            else:
                df = pd.read_excel(dest)
        except Exception as e:
            # surface a helpful error
            raise RuntimeError(f"Failed to read uploaded file '{dest.name}': {e}")

        # compute a stable fingerprint of the uploaded file so identical uploads
        # can share caches. Use SHA1 of raw file bytes.
        try:
            with open(dest, 'rb') as fh:
                file_bytes = fh.read()
            source_hash = hashlib.sha1(file_bytes).hexdigest()
        except Exception:
            source_hash = 'unknown'

        # Use a shared cache location under server_data/shared_cache/<source_hash>/<frequency>
        shared_cache = WORKDIR / 'shared_cache' / source_hash
        shared_cache.mkdir(parents=True, exist_ok=True)

        # model version can be provided via env var to allow invalidating caches when code changes
        model_version = os.getenv('FORECAST_MODEL_VERSION', 'v1')

        forecaster = SalesForecaster(
            dataframe=df, 
            cache_dir=str(shared_cache), 
            source_hash=source_hash, 
            model_version=model_version,
            frequency=frequency.lower()
        )
        # If the uploaded dataframe already contains aggregated summary fields
        # (e.g., 'avg_forecast' and 'trend_pct'), treat it as a precomputed summary
        # and avoid running clean_data()/prepare_data() which expect raw sales columns
        lower_cols = [c.lower() for c in df.columns]
        if 'avg_forecast' in lower_cols and 'trend_pct' in lower_cols:
            # Normalize column names to match forecaster expectations where possible
            # Ensure the reference column exists under forecaster.ref_col; otherwise try common variants
            # The SalesForecaster already attempts basic normalization in __init__
            try:
                # Assign the provided dataframe directly as grouped_data (summary)
                forecaster.grouped_data = df.copy()
                forecaster.df_clean = df.copy()
            except Exception:
                # Fall back to normal flow if assignment fails
                forecaster.clean_data()
                forecaster.prepare_data()
        else:
            forecaster.clean_data()
            forecaster.prepare_data()

        info = SessionInfo(sid, str(dest), forecaster)
        self._sessions[sid] = info
        return info

    def get(self, session_id: str) -> Optional[SessionInfo]:
        return self._sessions.get(session_id)

    def list_sessions(self):
        return list(self._sessions.keys())

    def delete_session(self, session_id: str) -> bool:
        """Remove session from registry and delete on-disk files."""
        info = self._sessions.get(session_id)
        if not info:
            return False
        # remove files on disk
        try:
            base = WORKDIR / session_id
            if base.exists():
                shutil.rmtree(base)
        except Exception:
            pass

        # remove from registry
        try:
            del self._sessions[session_id]
        except KeyError:
            pass
        return True


# singleton
REGISTRY = Registry()
