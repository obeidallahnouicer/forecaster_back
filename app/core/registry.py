from pathlib import Path
import uuid
import shutil
import pandas as pd
from typing import Dict, Optional
from sales_forecaster import SalesForecaster

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

    def create_session_from_file(self, file_path: str) -> SessionInfo:
        sid = str(uuid.uuid4())
        out_dir = WORKDIR / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / Path(file_path).name
        shutil.copy(file_path, dest)

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

        cache_dir = str(out_dir / "cache")
        forecaster = SalesForecaster(dataframe=df, cache_dir=cache_dir)
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
