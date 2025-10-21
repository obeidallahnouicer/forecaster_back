"""
Dataset Analyzer — Strict data-aware analysis to prevent hallucinations.

This module provides:
- Actual CSV data analysis (no estimates or guesses)
- Product stability calculations (based on trend_pct)
- Low-performing product identification
- Metrics validation against real data
- Dataset-aware response generation
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

from . import config, data_loader

logger = logging.getLogger("rag.dataset_analyzer")


# ============================================================================
# ANALYSIS THRESHOLDS — All values computed dynamically from data, these are
# minimum requirements for data quality, not derived from hardcoded examples.
# ============================================================================

# Minimum data points required for "reliable" analysis (prefer products with more data)
MIN_DATA_POINTS_FOR_RELIABILITY = 3

# Threshold for identifying "few datapoints" (< this value indicates low confidence)
FEW_DATA_POINTS_THRESHOLD = 5

# Quantile for identifying "low forecast" (lower than this percentile = low-performing)
LOW_FORECAST_QUANTILE = 0.25


class DatasetAnalyzer:
    """Analyzer for strict, data-backed reasoning without hallucinations."""
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize analyzer with the CSV data.
        
        Args:
            df: DataFrame to analyze. If None, loads from config.DATA_PATH
        """
        if df is None:
            logger.info("Loading dataset from config.DATA_PATH")
            self.df = data_loader.load_and_preprocess()
        else:
            logger.info(f"Using provided DataFrame with {len(df)} rows")
            self.df = df
        
        # Ensure key columns exist
        self._validate_columns()
        
        # Cache analysis results
        self._cache = {}
    
    def _validate_columns(self):
        """Ensure critical columns are present."""
        # Canonical required columns used by the analysis pipeline
        required = ["ref_article", "avg_forecast", "trend_pct", "trend_label", "data_points"]

        # Build a simple normalization map of common variant headers to canonical names.
        # This helps handle CSVs with different naming conventions (e.g., 'Ref Article', 'CA HT NET').
        variants_map = {
            "ref_article": ["ref_article", "ref article", "ref-article", "reference", "article_ref", "ref"],
            "avg_forecast": ["avg_forecast", "avg forecast", "ca ht net", "ca_ht_net", "ca_ht", "forecast", "forecast_next_year", "next_year"],
            "trend_pct": ["trend_pct", "trend pct", "trend%", "trend_percent", "pct_change", "trend"],
            "trend_label": ["trend_label", "trend label", "trend_class", "trend_label_clean"],
            "data_points": ["data_points", "data points", "n_points", "count", "observations", "num_points"],
        }

        # Lowercase current columns for matching while preserving original names for renaming
        orig_cols = list(self.df.columns)
        lowered = [c.lower().strip() for c in orig_cols]

        rename_map = {}
        for canon, variants in variants_map.items():
            for v in variants:
                if v in lowered:
                    # find original column name that matches this lowered token
                    idx = lowered.index(v)
                    original_name = orig_cols[idx]
                    # Only rename if original differs from canonical to avoid unnecessary ops
                    if original_name != canon:
                        rename_map[original_name] = canon
                    break

        if rename_map:
            try:
                self.df = self.df.rename(columns=rename_map)
                logger.info(f"Normalized dataset columns using mapping: {rename_map}")
                # Refresh orig_cols/lowered after rename
                orig_cols = list(self.df.columns)
                lowered = [c.lower().strip() for c in orig_cols]
            except Exception:
                logger.exception("Failed to rename dataset columns during normalization")

        # Normalize column names to lowercase for consistent access throughout the analyzer
        try:
            self.df.columns = [c.lower() for c in self.df.columns]
        except Exception:
            # If column normalization fails, surface it — callers expect analyzer.df to be usable
            logger.exception("Failed to normalize DataFrame column names to lowercase")

        # Compute missing required columns after normalization
        missing = [c for c in required if c not in [x.lower() for x in self.df.columns]]
        if missing:
            logger.warning(f"Missing columns: {missing}. Working with available columns.")
    
    def get_all_products(self) -> pd.DataFrame:
        """Return all products in the dataset."""
        return self.df.copy()
    
    def get_product_count(self) -> int:
        """Return total number of unique products."""
        if "ref_article" in self.df.columns:
            return len(self.df["ref_article"].unique())
        return len(self.df)
    
    def find_products_by_trend(self, trend_label: str) -> pd.DataFrame:
        """Find all products with a specific trend label.
        
        Args:
            trend_label: "Uptrend", "Downtrend", or "Stable"
            
        Returns:
            DataFrame of matching products
        """
        if "trend_label" not in self.df.columns:
            return pd.DataFrame()
        
        matches = self.df[self.df["trend_label"].str.lower() == trend_label.lower()]
        return matches
    
    def find_most_stable_product(self) -> Optional[Dict[str, Any]]:
        """
        Find the most stable product (smallest absolute trend_pct, with preference for products with more data points).
        
        Logic:
        1. Filter products with trend_pct closest to zero (most stable)
        2. Among products with similar stability, prefer those with more data points (more reliable)
        3. Ignore products with only 1-2 data points (low confidence)
        
        Returns:
            Dict with product info or None if not found
        """
        if "trend_pct" not in self.df.columns or "ref_article" not in self.df.columns:
            return None
        
        # Filter out NaN trend_pct values and work on a copy
        df_valid = self.df[self.df["trend_pct"].notna()].copy()

        if df_valid.empty:
            return None

        # Coerce numeric columns to proper types to avoid unexpected comparisons
        try:
            df_valid["trend_pct"] = pd.to_numeric(df_valid["trend_pct"], errors="coerce")
        except Exception:
            logger.debug("Could not coerce 'trend_pct' to numeric; leaving as-is")

        if "data_points" in df_valid.columns:
            # convert to integer, fill NaN with 0
            df_valid["data_points"] = pd.to_numeric(df_valid["data_points"], errors="coerce").fillna(0).astype(int)
        else:
            # If the column is missing, create it as zeros so comparisons behave deterministically
            df_valid["data_points"] = 0

        # Primary: products with at least MIN_DATA_POINTS_FOR_RELIABILITY
        df_reliable = df_valid[df_valid["data_points"] >= MIN_DATA_POINTS_FOR_RELIABILITY].copy()

        # Do NOT fallback to lower-data candidates. If no reliable products are available,
        # refuse to pick a most-stable product to avoid making decisions from insufficient data.
        if df_reliable.empty:
            logger.warning(
                "No products meet MIN_DATA_POINTS_FOR_RELIABILITY for reliable stability selection; returning None"
            )
            return None

        # Compute absolute trend for ranking and apply tie-breakers: prefer more data points, then larger avg_forecast
        df_reliable["abs_trend"] = df_reliable["trend_pct"].abs()

        # Ensure avg_forecast exists and is numeric for tie-breaker
        if "avg_forecast" in df_reliable.columns:
            df_reliable["avg_forecast"] = pd.to_numeric(df_reliable["avg_forecast"], errors="coerce").fillna(0)
        else:
            df_reliable["avg_forecast"] = 0

        # Sort by abs_trend asc (most stable), data_points desc (more reliable), avg_forecast desc
        df_sorted = df_reliable.sort_values(by=["abs_trend", "data_points", "avg_forecast"], ascending=[True, False, False])

        if df_sorted.empty:
            return None

        row = df_sorted.iloc[0]

        return {
            "product": row["ref_article"],
            "avg_forecast": float(row.get("avg_forecast", 0)),
            "trend_pct": float(row.get("trend_pct", 0)),
            "trend_label": row.get("trend_label", "Unknown"),
            "data_points": int(row.get("data_points", 0)),
            "designation": row.get("designation", ""),
        }
    
    def find_low_performing_product(self) -> Optional[Dict[str, Any]]:
        """
        Find the worst-performing product based on:
        1. Lowest avg_forecast
        2. Negative trend_pct or Downtrend label
        3. Fewest data_points
        
        Returns:
            Dict with product info or None if not found
        """
        if "avg_forecast" not in self.df.columns or "ref_article" not in self.df.columns:
            return None

        df_valid = self.df[self.df["avg_forecast"].notna()].copy()

        if df_valid.empty:
            return None

        # Candidates must meet minimum data-point reliability to be considered for removal.
        df_candidates = df_valid[df_valid.get("data_points", 0) >= MIN_DATA_POINTS_FOR_RELIABILITY].copy()
        if df_candidates.empty:
            logger.warning(
                "No products meet MIN_DATA_POINTS_FOR_RELIABILITY for low-performing selection; returning None"
            )
            return None

        # Score products: lower avg_forecast, negative trend, fewer data points
        df_candidates["downtrend"] = 0
        if "trend_pct" in df_candidates.columns:
            df_candidates["downtrend"] = df_candidates["downtrend"] + (df_candidates["trend_pct"] < 0).astype(int) * 10
        if "trend_label" in df_candidates.columns:
            df_candidates["downtrend"] = df_candidates["downtrend"] + (df_candidates["trend_label"].str.lower() == "downtrend").astype(int) * 5

        df_candidates["few_datapoints"] = 0
        if "data_points" in df_candidates.columns:
            df_candidates["few_datapoints"] = (df_candidates["data_points"] < FEW_DATA_POINTS_THRESHOLD).astype(int) * 3

        # Normalize avg_forecast and create score
        min_forecast = df_candidates["avg_forecast"].min()
        max_forecast = df_candidates["avg_forecast"].max()
        if max_forecast > min_forecast:
            df_candidates["forecast_score"] = (
                (max_forecast - df_candidates["avg_forecast"]) / (max_forecast - min_forecast) * 100
            )
        else:
            df_candidates["forecast_score"] = 0

        df_candidates["total_score"] = (
            df_candidates["forecast_score"] * 0.5 +
            df_candidates["downtrend"] * 0.3 +
            df_candidates["few_datapoints"] * 0.2
        )

        # Find worst
        idx = df_candidates["total_score"].idxmax()
        row = df_candidates.loc[idx]

        return {
            "product": row["ref_article"],
            "avg_forecast": float(row.get("avg_forecast", 0)),
            "trend_pct": float(row.get("trend_pct", 0)),
            "trend_label": row.get("trend_label", "Unknown"),
            "data_points": int(row.get("data_points", 0)),
            "designation": row.get("designation", ""),
            "reason": self._format_low_performing_reason(row),
        }
    
    def _format_low_performing_reason(self, row: pd.Series) -> str:
        """Format reason why a product is low-performing."""
        reasons = []
        
        # Check avg_forecast (use LOW_FORECAST_QUANTILE to define threshold)
        if row.get("avg_forecast", 0) < self.df["avg_forecast"].quantile(LOW_FORECAST_QUANTILE):
            reasons.append(f"lowest average forecast ({row['avg_forecast']:.2f})")
        
        # Check trend
        trend_pct = row.get("trend_pct", 0)
        if trend_pct < 0:
            reasons.append(f"downtrend of {trend_pct:.2f}%")
        
        # Check data points
        if row.get("data_points", 0) < FEW_DATA_POINTS_THRESHOLD:
            reasons.append(f"only {int(row['data_points'])} data points")
        
        if not reasons:
            return "mixed performance indicators"
        
        return " and ".join(reasons)
    
    def get_top_products_by_forecast(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N products by average forecast."""
        if "avg_forecast" not in self.df.columns or "ref_article" not in self.df.columns:
            return []
        
        df_sorted = self.df.nlargest(n, "avg_forecast")
        
        results = []
        for _, row in df_sorted.iterrows():
            results.append({
                "product": row["ref_article"],
                "avg_forecast": float(row.get("avg_forecast", 0)),
                "trend_pct": float(row.get("trend_pct", 0)),
                "trend_label": row.get("trend_label", "Unknown"),
                "data_points": int(row.get("data_points", 0)),
            })
        
        return results
    
    def get_products_by_uptrend(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get products with Uptrend label."""
        if "trend_label" not in self.df.columns:
            return []
        
        uptrend_df = self.df[self.df["trend_label"].str.lower() == "uptrend"]
        if uptrend_df.empty:
            return []
        
        # Sort by avg_forecast descending
        uptrend_df = uptrend_df.nlargest(limit, "avg_forecast")
        
        results = []
        for _, row in uptrend_df.iterrows():
            results.append({
                "product": row["ref_article"],
                "avg_forecast": float(row.get("avg_forecast", 0)),
                "trend_pct": float(row.get("trend_pct", 0)),
                "trend_label": "Uptrend",
                "data_points": int(row.get("data_points", 0)),
                "designation": row.get("designation", ""),
            })
        
        return results
    
    def verify_product_exists(self, product_code: str) -> bool:
        """Check if a product exists in the dataset."""
        if "ref_article" not in self.df.columns:
            return False
        
        return product_code in self.df["ref_article"].values
    
    def get_product_info(self, product_code: str) -> Optional[Dict[str, Any]]:
        """Get all available info for a specific product."""
        if "ref_article" not in self.df.columns:
            return None
        
        match = self.df[self.df["ref_article"] == product_code]
        
        if match.empty:
            return None
        
        row = match.iloc[0]
        
        return {
            "product": row["ref_article"],
            "designation": row.get("designation", ""),
            "marque": row.get("marque", ""),
            "famille": row.get("famille", ""),
            "avg_forecast": float(row.get("avg_forecast", 0)),
            "trend_pct": float(row.get("trend_pct", 0)),
            "trend_label": row.get("trend_label", ""),
            "data_points": int(row.get("data_points", 0)),
            "next_year": row.get("next_year", ""),
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the dataset."""
        stats = {
            "total_products": int(self.get_product_count()),
        }
        
        if "avg_forecast" in self.df.columns:
            stats["avg_forecast_mean"] = float(self.df["avg_forecast"].mean())
            stats["avg_forecast_min"] = float(self.df["avg_forecast"].min())
            stats["avg_forecast_max"] = float(self.df["avg_forecast"].max())
        
        trend_dist = {"uptrend": 0, "downtrend": 0, "stable": 0}
        if "trend_label" in self.df.columns:
            trend_dist["uptrend"] = int((self.df["trend_label"] == "Uptrend").sum())
            trend_dist["downtrend"] = int((self.df["trend_label"] == "Downtrend").sum())
            trend_dist["stable"] = int((self.df["trend_label"] == "Stable").sum())
        
        stats["trend_distribution"] = trend_dist
        return stats


# Global instance (lazy loaded)
_analyzer: Optional[DatasetAnalyzer] = None
_analyzer_lock = None


def get_analyzer(df: Optional[pd.DataFrame] = None) -> DatasetAnalyzer:
    """Get or create a global DatasetAnalyzer instance."""
    global _analyzer, _analyzer_lock
    
    if _analyzer_lock is None:
        import threading
        _analyzer_lock = threading.Lock()
    
    with _analyzer_lock:
        if _analyzer is None:
            # If caller didn't provide a DataFrame, prefer any uploaded server files in server_data/
            if df is None:
                repo_root = Path(__file__).resolve().parent.parent
                # If forced to use repo-level summary, load that only
                try:
                    from . import config as _cfg
                except Exception:
                    _cfg = None

                if _cfg and getattr(_cfg, 'FORCE_REPO_SUMMARY', False):
                    fallback = repo_root / "forecast-summary.csv"
                    if fallback.exists():
                        logger.info(f"FORCE_REPO_SUMMARY enabled: loading {fallback}")
                        df_try = data_loader.load_and_preprocess(path=fallback)
                        _analyzer = DatasetAnalyzer(df_try)
                    else:
                        raise FileNotFoundError("FORCE_REPO_SUMMARY is set but forecast-summary.csv not found at repo root")
                else:
                    # 1) Check for server-side uploads (server_data/<session_id>/*)
                    server_dir = repo_root / "server_data"
                    try:
                        if server_dir.exists() and any(server_dir.iterdir()):
                            # Find newest file under server_data (csv/xlsx)
                            candidates = list(server_dir.glob("**/*.*"))
                            candidates = [p for p in candidates if p.suffix.lower() in ('.csv', '.xlsx', '.xls')]
                            if candidates:
                                newest = max(candidates, key=lambda p: p.stat().st_mtime)
                                logger.info(f"Initializing DatasetAnalyzer from latest uploaded file: {newest}")
                                df_try = data_loader.load_and_preprocess(path=newest)
                                _analyzer = DatasetAnalyzer(df_try)
                            else:
                                raise FileNotFoundError("No uploaded CSV/XLSX files found under server_data")
                        else:
                            raise FileNotFoundError("server_data directory missing or empty")
                    except Exception:
                        # 2) Fallback to repo-root forecast-summary.csv
                        try:
                            fallback = repo_root / "forecast-summary.csv"
                            if fallback.exists():
                                logger.info(f"Falling back to repo-root forecast-summary.csv: {fallback}")
                                df_try = data_loader.load_and_preprocess(path=fallback)
                                _analyzer = DatasetAnalyzer(df_try)
                            else:
                                # 3) Finally try configured DATA_PATH via default DatasetAnalyzer init
                                logger.info("No uploaded file or repo-level forecast-summary.csv found; using config.DATA_PATH")
                                _analyzer = DatasetAnalyzer(None)
                        except Exception:
                            logger.exception("Unable to create DatasetAnalyzer from server_data, repo fallback, or config.DATA_PATH")
                            _analyzer = DatasetAnalyzer(None)
            else:
                _analyzer = DatasetAnalyzer(df)
        return _analyzer


def reset_analyzer():
    """Reset the global analyzer instance (useful for testing)."""
    global _analyzer
    _analyzer = None
