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
        required = ["ref_article", "avg_forecast", "trend_pct", "trend_label", "data_points"]
        missing = [c for c in required if c.lower() not in [x.lower() for x in self.df.columns]]
        
        if missing:
            logger.warning(f"Missing columns: {missing}. Working with available columns.")
        
        # Normalize column names to lowercase for consistent access
        self.df.columns = [c.lower() for c in self.df.columns]
    
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
        
        # Filter out NaN values and products with very few data points
        df_valid = self.df[self.df["trend_pct"].notna()].copy()
        
        if df_valid.empty:
            return None
        
        # Prefer products with more data points (at least MIN_DATA_POINTS_FOR_RELIABILITY) for higher confidence
        df_reliable = df_valid[df_valid.get("data_points", 0) >= MIN_DATA_POINTS_FOR_RELIABILITY].copy()
        
        if df_reliable.empty:
            # Fallback to all products if none have enough data points
            df_reliable = df_valid.copy()
        
        # Find row with smallest absolute trend_pct
        df_reliable["abs_trend"] = df_reliable["trend_pct"].abs()
        idx = df_reliable["abs_trend"].idxmin()
        row = df_reliable.loc[idx]
        
        return {
            "product": row["ref_article"],
            "avg_forecast": float(row.get("avg_forecast", 0)),
            "trend_pct": float(row["trend_pct"]),
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
        
        # Score products: lower avg_forecast, negative trend, fewer data points
        df_valid["downtrend"] = 0
        if "trend_pct" in df_valid.columns:
            df_valid["downtrend"] = df_valid["downtrend"] + (df_valid["trend_pct"] < 0).astype(int) * 10
        if "trend_label" in df_valid.columns:
            df_valid["downtrend"] = df_valid["downtrend"] + (df_valid["trend_label"].str.lower() == "downtrend").astype(int) * 5
        
        df_valid["few_datapoints"] = 0
        if "data_points" in df_valid.columns:
            df_valid["few_datapoints"] = (df_valid["data_points"] < FEW_DATA_POINTS_THRESHOLD).astype(int) * 3
        
        # Normalize avg_forecast and create score
        min_forecast = df_valid["avg_forecast"].min()
        max_forecast = df_valid["avg_forecast"].max()
        if max_forecast > min_forecast:
            df_valid["forecast_score"] = (
                (max_forecast - df_valid["avg_forecast"]) / (max_forecast - min_forecast) * 100
            )
        else:
            df_valid["forecast_score"] = 0
        
        df_valid["total_score"] = (
            df_valid["forecast_score"] * 0.5 +
            df_valid["downtrend"] * 0.3 +
            df_valid["few_datapoints"] * 0.2
        )
        
        # Find worst
        idx = df_valid["total_score"].idxmax()
        row = df_valid.loc[idx]
        
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
            _analyzer = DatasetAnalyzer(df)
        return _analyzer


def reset_analyzer():
    """Reset the global analyzer instance (useful for testing)."""
    global _analyzer
    _analyzer = None
