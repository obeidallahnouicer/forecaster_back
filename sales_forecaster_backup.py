import os
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import unicodedata
import difflib

# Try imports that may be optional
try:
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_ARIMA = True
except Exception:
    _HAS_ARIMA = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

warnings.filterwarnings("ignore")


class SalesForecaster:
    """
    Enhanced SalesForecaster with:
      - Monthly AND Yearly forecasting support
      - Detailed metrics for each forecasting method
      - Per-article model and forecast caching
      - Fast-mode heuristics for small series
    """

    def __init__(self, dataframe: pd.DataFrame,
                 cache_dir: str = "cache",
                 ref_col: str = "Ref Article",
                 date_col: str = "Année",  # Can be year or date
                 sales_col: str = "CA HT NET",
                 frequency: str = "yearly",  # 'yearly' or 'monthly'
                 source_hash: str = None,
                 model_version: str = "v1"):
        """
        dataframe: raw dataframe containing at least [ref_col, date_col, sales_col]
        cache_dir: folder to store cached models/forecasts/summaries
        frequency: 'yearly' or 'monthly' for aggregation level
        source_hash: optional fingerprint for cache sharing
        model_version: version identifier for model cache invalidation
        """
        self.df_raw = dataframe.copy()
        self.ref_col = ref_col
        self.date_col = date_col
        self.sales_col = sales_col
        self.frequency = frequency.lower()

        # ensure standard columns rename when necessary
        self._normalize_column_names()

        self.df_clean = None
        self.grouped_data = None
        self.forecast_results = None

        # cache structure - use shared cache if source_hash provided
        if source_hash:
            # Shared cache: cache_dir/source_hash/frequency/
            self.cache_dir = Path(cache_dir) / source_hash / self.frequency
        else:
            # Regular cache: cache_dir/frequency/
            self.cache_dir = Path(cache_dir) / self.frequency
        
        self.model_cache = self.cache_dir / "models"
        self.forecast_cache = self.cache_dir / "forecasts"
        self.summary_cache = self.cache_dir / "summary"
        for p in (self.model_cache, self.forecast_cache, self.summary_cache):
            p.mkdir(parents=True, exist_ok=True)

        # availability flags
        self.has_arima = _HAS_ARIMA
        self.has_prophet = _HAS_PROPHET
        self.has_xgboost = _HAS_XGBOOST

        # source fingerprint to tie caches to exact uploaded dataset
        # If not provided, derive from dataframe content (deterministic)
        self.source_hash = source_hash
        if not self.source_hash:
            try:
                # Use a stable CSV serialization to compute hash
                buf = self.df_raw.to_csv(index=False).encode('utf-8')
                import hashlib
                self.source_hash = hashlib.sha1(buf).hexdigest()
            except Exception:
                self.source_hash = "unknown"

        # model_version used to invalidate caches when forecasting logic or models change
        self.model_version = model_version

    def _normalize_column_names(self):
        # Basic renames preserved from legacy datasets
        rename_map = {
            'Intitule Marque': 'Marque',
            'Intitule Famille': 'Famille',
            'Intitule Sous Famille': 'Sous Famille'
        }
        for old, new in rename_map.items():
            if old in self.df_raw.columns and new not in self.df_raw.columns:
                self.df_raw.rename(columns={old: new}, inplace=True)

        # Additional normalization: map common variants of the key columns (ref, sales, date)
        # to the expected names (self.ref_col, self.sales_col, self.date_col).
        # Use unicode normalization and fuzzy matching to be robust to accents and punctuation.

        def norm(s: str) -> str:
            if s is None:
                return ""
            # Normalize unicode accents, lowercase, replace punctuation with spaces, collapse spaces
            s = str(s)
            s = unicodedata.normalize('NFKD', s)
            s = ''.join(ch for ch in s if not unicodedata.combining(ch))
            s = s.lower()
            # Replace non-alphanumeric with spaces
            s = ''.join(ch if ch.isalnum() else ' ' for ch in s)
            s = ' '.join(s.split())
            return s

        existing = list(self.df_raw.columns)
        normalized_map = {norm(c): c for c in existing}

        variants = {
            'ref': [
                'ref article', 'ref_article', 'ref-article', 'reference', 'article ref', 'ref'
            ],
            'sales': [
                'ca ht net', 'ca_ht_net', 'cahtnet', 'sales', 'sales ht net', 'amount ht', 'ca ht', 'net sales', 'montant ht', "chiffre d'affaire", 'chiffre affaires', 'chiffre affaires net'
            ],
            'date': [
                'annee', 'année', 'year', 'date', 'periode', 'period', 'period key'
            ]
        }

        import logging
        logger_sf = logging.getLogger('sales_forecaster')

        def find_and_rename(target_col_name: str, variant_list: list):
            # Already present
            if target_col_name in self.df_raw.columns:
                return
            # Try exact normalized match first
            for v in variant_list:
                vnorm = norm(v)
                if vnorm in normalized_map:
                    orig = normalized_map[vnorm]
                    try:
                        self.df_raw.rename(columns={orig: target_col_name}, inplace=True)
                        logger_sf.info(f"Normalized column: '{orig}' -> '{target_col_name}'")
                        # update maps
                        normalized_map[norm(target_col_name)] = target_col_name
                        if vnorm in normalized_map:
                            del normalized_map[vnorm]
                    except Exception:
                        logger_sf.exception(f"Failed to rename column {orig} to {target_col_name}")
                    return

            # No exact normalized match; try fuzzy match among normalized names
            choices = list(normalized_map.keys())
            for v in variant_list:
                vnorm = norm(v)
                matches = difflib.get_close_matches(vnorm, choices, n=2, cutoff=0.8)
                if matches:
                    orig = normalized_map[matches[0]]
                    try:
                        self.df_raw.rename(columns={orig: target_col_name}, inplace=True)
                        logger_sf.info(f"Fuzzy-normalized column: '{orig}' -> '{target_col_name}' (matched '{v}')")
                        normalized_map[norm(target_col_name)] = target_col_name
                        if matches[0] in normalized_map:
                            del normalized_map[matches[0]]
                    except Exception:
                        logger_sf.exception(f"Failed fuzzy rename column {orig} to {target_col_name}")
                    return

        # Apply normalization for expected column names
        find_and_rename(self.ref_col, variants['ref'])
        find_and_rename(self.sales_col, variants['sales'])
        find_and_rename(self.date_col, variants['date'])

    # -------------------------
    # Data prep with monthly/yearly support
    # -------------------------
    def clean_data(self):
        """Remove rows where sales are null or zero and keep relevant columns"""
        df = self.df_raw
        required = [self.ref_col, self.sales_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            # Last-resort: try to find close matches among existing columns using unicode normalization
            def norm_local(s: str) -> str:
                s = str(s)
                s = unicodedata.normalize('NFKD', s)
                s = ''.join(ch for ch in s if not unicodedata.combining(ch))
                s = s.lower()
                s = ''.join(ch if ch.isalnum() else ' ' for ch in s)
                s = ' '.join(s.split())
                return s

            existing = list(df.columns)
            existing_norm = {norm_local(c): c for c in existing}

            import difflib
            attempted = {}
            for req_col in list(missing):
                target_norm = norm_local(req_col)
                # exact normalized match
                if target_norm in existing_norm:
                    orig = existing_norm[target_norm]
                    df.rename(columns={orig: req_col}, inplace=True)
                    attempted[req_col] = orig
                    missing.remove(req_col)
                    continue

                # fuzzy matches
                choices = list(existing_norm.keys())
                matches = difflib.get_close_matches(target_norm, choices, n=1, cutoff=0.7)
                if matches:
                    orig = existing_norm[matches[0]]
                    df.rename(columns={orig: req_col}, inplace=True)
                    attempted[req_col] = orig
                    missing.remove(req_col)

            if attempted:
                try:
                    import logging
                    logging.getLogger('sales_forecaster').info(f"Applied fallback renames in clean_data: {attempted}")
                except Exception:
                    pass

            if missing:
                # Provide a clearer error message with available columns and suggestions
                available = existing
                suggestions = {}
                for req_col in missing:
                    target_norm = norm_local(req_col)
                    choices = list(existing_norm.keys())
                    close = difflib.get_close_matches(target_norm, choices, n=3, cutoff=0.5)
                    suggestions[req_col] = [existing_norm[c] for c in close]

                raise ValueError(
                    f"Missing columns in the dataset: {missing}. Available columns: {available}. "
                    f"Suggestions: {suggestions}"
                )

        df_clean = df[df[self.sales_col].notna() & (df[self.sales_col] != 0)].copy()
        
        # Handle date column based on frequency
        if self.frequency == 'yearly':
            # Use Année column
            if 'Année' not in df_clean.columns:
                raise ValueError("'Année' column not found for yearly frequency")
            df_clean['period_key'] = df_clean['Année'].astype(int)
        else:  # monthly
            # Parse the Date column to extract year-month
            if 'Date' not in df_clean.columns:
                raise ValueError("'Date' column not found for monthly frequency")
            if not pd.api.types.is_datetime64_any_dtype(df_clean['Date']):
                df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            df_clean = df_clean[df_clean['Date'].notna()]
            # Create year-month period
            df_clean['period_key'] = df_clean['Date'].dt.to_period('M')
            
        self.df_clean = df_clean
        return self.df_clean

    def prepare_data(self):
        """Group by article and period (year or month) summing sales"""
        if self.df_clean is None:
            self.clean_data()

        self.df_clean[self.ref_col] = self.df_clean[self.ref_col].astype(str)

        available_columns = [c for c in ['Marque', 'Famille', 'Sous Famille', 'Designation'] 
                           if c in self.df_clean.columns]
        agg_dict = {self.sales_col: 'sum'}
        for col in available_columns:
            agg_dict[col] = 'first'

        grouped = self.df_clean.groupby([self.ref_col, 'period_key']).agg(agg_dict).reset_index()
        grouped.rename(columns={'period_key': 'period'}, inplace=True)
        grouped[self.ref_col] = grouped[self.ref_col].astype(str)
        grouped = grouped.sort_values([self.ref_col, 'period']).reset_index(drop=True)

        self.grouped_data = grouped
        return grouped

    # -------------------------
    # Metric calculation utilities
    # -------------------------
    def calculate_metrics(self, actual, predicted):
        """Calculate comprehensive metrics for a forecast method"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        if len(actual) == 0 or len(predicted) == 0:
            return None
            
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mask = actual != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            mape = np.nan
            
        # R² score
        try:
            r2 = r2_score(actual, predicted)
        except:
            r2 = np.nan
        
        return {
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'R2': float(r2)
        }

    # -------------------------
    # Low-cost forecasting helpers with metrics
    # -------------------------
    def simple_moving_average(self, values, period=3, return_metrics=False):
        if len(values) == 0:
            return (0.0, None) if return_metrics else 0.0
        
        if len(values) < period:
            forecast = float(np.mean(values))
        else:
            forecast = float(np.mean(values[-period:]))
        
        if return_metrics and len(values) > period:
            # Calculate metrics on historical predictions
            predictions = []
            actuals = []
            for i in range(period, len(values)):
                pred = np.mean(values[i-period:i])
                predictions.append(pred)
                actuals.append(values[i])
            metrics = self.calculate_metrics(actuals, predictions)
        else:
            metrics = None
            
        return (forecast, metrics) if return_metrics else forecast

    def exponential_smoothing(self, values, alpha=0.3, return_metrics=False):
        if len(values) == 0:
            return (0.0, None) if return_metrics else 0.0
        if len(values) == 1:
            return (float(values[0]), None) if return_metrics else float(values[0])
        
        # Calculate forecast and predictions for metrics
        predictions = []
        f = values[0]
        for i, v in enumerate(values[1:], 1):
            predictions.append(f)
            f = alpha * v + (1 - alpha) * f
        
        forecast = alpha * values[-1] + (1 - alpha) * f
        
        if return_metrics and len(predictions) > 0:
            metrics = self.calculate_metrics(values[1:], predictions)
        else:
            metrics = None
            
        return (float(forecast), metrics) if return_metrics else float(forecast)

    def linear_regression_forecast(self, periods, values, return_metrics=False):
        if len(periods) == 0:
            return (0.0, None) if return_metrics else 0.0
        
        # Convert periods to numeric
        if self.frequency == 'monthly':
            X = np.array([p.ordinal for p in periods]).reshape(-1, 1)
            next_period = periods[-1] + 1
            next_X = np.array([[next_period.ordinal]])
        else:
            X = np.array([int(p) for p in periods]).reshape(-1, 1)
            next_X = np.array([[int(max(periods) + 1)]])
        
        y = np.array(values)
        model = LinearRegression()
        model.fit(X, y)
        
        forecast = float(model.predict(next_X)[0])
        forecast = max(0.0, forecast)
        
        if return_metrics:
            predictions = model.predict(X)
            metrics = self.calculate_metrics(y, predictions)
        else:
            metrics = None
            
        return (forecast, metrics) if return_metrics else forecast

    # -------------------------
    # Time series / ML forecasts with metrics
    # -------------------------
    def arima_forecast(self, values, order=(1, 1, 0), return_metrics=False):
        if len(values) == 0 or not self.has_arima:
            return (0.0, None) if return_metrics else 0.0
        try:
            model = ARIMA(values, order=order)
            fit = model.fit()
            forecast = float(fit.forecast(steps=1)[0])
            forecast = max(0.0, forecast)
            
            if return_metrics and len(values) > 3:
                # In-sample predictions
                predictions = fit.fittedvalues
                if len(predictions) == len(values):
                    metrics = self.calculate_metrics(values, predictions)
                else:
                    metrics = None
            else:
                metrics = None
                
            return (forecast, metrics) if return_metrics else forecast
        except Exception:
            return (0.0, None) if return_metrics else 0.0

    def prophet_forecast(self, periods, values, return_metrics=False):
        if len(values) < 2 or not self.has_prophet:
            return (0.0, None) if return_metrics else 0.0
        try:
            # Convert periods to datetime
            if self.frequency == 'monthly':
                dates = [p.to_timestamp() for p in periods]
            else:
                dates = pd.to_datetime([str(int(p)) for p in periods], format='%Y')
            
            dfp = pd.DataFrame({'ds': dates, 'y': values})
            m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
            m.fit(dfp)
            
            freq = 'MS' if self.frequency == 'monthly' else 'YS'
            future = m.make_future_dataframe(periods=1, freq=freq)
            fc = m.predict(future)
            forecast = float(max(0.0, fc.iloc[-1]['yhat']))
            
            if return_metrics:
                in_sample = m.predict(dfp)
                metrics = self.calculate_metrics(values, in_sample['yhat'].values)
            else:
                metrics = None
                
            return (forecast, metrics) if return_metrics else forecast
        except Exception as e:
            return (0.0, None) if return_metrics else 0.0

    def xgboost_forecast(self, periods, values, return_metrics=False):
        if len(values) == 0:
            return (0.0, None) if return_metrics else 0.0
        
        # Convert periods to numeric
        if self.frequency == 'monthly':
            X = np.array([p.ordinal for p in periods]).reshape(-1, 1)
            next_period = periods[-1] + 1
            next_X = np.array([[next_period.ordinal]])
        else:
            X = np.array([int(p) for p in periods]).reshape(-1, 1)
            next_X = np.array([[int(max(periods) + 1)]])
        
        y = np.array(values)
        
        try:
            if self.has_xgboost:
                model = XGBRegressor(n_estimators=200, learning_rate=0.05, verbosity=0, n_jobs=1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            forecast = float(model.predict(next_X)[0])
            forecast = max(0.0, forecast)
            
            if return_metrics:
                predictions = model.predict(X)
                metrics = self.calculate_metrics(y, predictions)
            else:
                metrics = None
                
            return (forecast, metrics) if return_metrics else forecast
        except Exception:
            return (0.0, None) if return_metrics else 0.0

    # -------------------------
    # Caching utilities
    # -------------------------
    def _model_cache_path(self, ref_article, model_name):
        safe = str(ref_article).replace("/", "_").replace(" ", "_")
        return self.model_cache / f"{safe}__{model_name}.pkl"

    def _forecast_cache_path(self, ref_article):
        safe = str(ref_article).replace("/", "_").replace(" ", "_")
        return self.forecast_cache / f"{safe}__forecast.csv"

    def _summary_cache_path(self):
        return self.summary_cache / "summary.parquet"

    # -------------------------
    # Per-article orchestration
    # -------------------------
    def _get_article_series(self, ref_article):
        """Return sorted (periods, values) arrays for the article"""
        if self.grouped_data is None:
            self.prepare_data()
        df = self.grouped_data[self.grouped_data[self.ref_col] == ref_article].sort_values('period')
        if df.empty:
            return [], [], {}
        
        periods = df['period'].tolist()
        values = df[self.sales_col].astype(float).tolist()
        
        # metadata
        metadata = {}
        for c in ['Designation', 'Marque', 'Famille']:
            if c in df.columns:
                metadata[c.lower()] = df[c].iloc[0]
            else:
                metadata[c.lower()] = None
        return periods, values, metadata

    def _load_forecast_cache(self, ref_article, use_cache=True):
        p = self._forecast_cache_path(ref_article)
        if use_cache and p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
        return None

    def _save_forecast_cache(self, ref_article, df):
        p = self._forecast_cache_path(ref_article)
        df.to_csv(p, index=False)

    # -------------------------
    # Main per-article forecast method with detailed metrics
    # -------------------------
    def forecast_article(self, ref_article, period=3, alpha=0.3,
                         force_recompute=False, include_methods=None,
                         fast_mode=True, return_metrics=True):
        """
        Forecast for a single article with detailed metrics for each method.
        Produces output compatible with both Streamlit dashboard and API consumers.
        """
        import json
        
        if include_methods is None:
            include_methods = ['SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST']

        cached = None if force_recompute else self._load_forecast_cache(ref_article, use_cache=True)
        if cached is not None:
            return cached.to_dict(orient='records')[0] if not cached.empty else None

        periods, values, meta = self._get_article_series(ref_article)
        if len(values) == 0:
            return None

        # Determine next period and legacy next_year
        if self.frequency == 'monthly':
            next_period = str(periods[-1] + 1)  # ✅ FIX: Convert to string for JSON serialization
            # next_year legacy: use year of next monthly period
            try:
                next_year = int((periods[-1] + 1).to_timestamp().year)
            except Exception:
                next_year = None
        else:
            next_period = int(max(periods) + 1)
            next_year = int(next_period)

        # Fast-mode heuristics
        if fast_mode and len(values) < 6:
            allowed = [m for m in include_methods if m in ['SMA', 'ExpSmoothing', 'LinearReg', 'XGBOOST']]
            include_methods = allowed

        # Compute forecasts and metrics
        results = {}
        metrics_dict = {}
        
        if 'SMA' in include_methods:
            forecast, metrics = self.simple_moving_average(values, period, return_metrics=True)
            results['sma_forecast'] = forecast
            metrics_dict['sma_metrics'] = metrics
        
        if 'ExpSmoothing' in include_methods:
            forecast, metrics = self.exponential_smoothing(values, alpha, return_metrics=True)
            results['es_forecast'] = forecast
            metrics_dict['es_metrics'] = metrics
        
        if 'LinearReg' in include_methods:
            forecast, metrics = self.linear_regression_forecast(periods, values, return_metrics=True)
            results['lr_forecast'] = forecast
            metrics_dict['lr_metrics'] = metrics
        
        if 'ARIMA' in include_methods and self.has_arima and len(values) >= 3:
            forecast, metrics = self.arima_forecast(values, return_metrics=True)
            results['arima_forecast'] = forecast
            metrics_dict['arima_metrics'] = metrics
        
        if 'PROPHET' in include_methods and self.has_prophet and len(values) >= 3:
            forecast, metrics = self.prophet_forecast(periods, values, return_metrics=True)
            results['prophet_forecast'] = forecast
            metrics_dict['prophet_metrics'] = metrics
        
        if 'XGBOOST' in include_methods:
            forecast, metrics = self.xgboost_forecast(periods, values, return_metrics=True)
            results['xgb_forecast'] = forecast
            metrics_dict['xgb_metrics'] = metrics

        # Build ensemble average
        method_values = [v for v in results.values() if v is not None and not np.isnan(v)]
        avg_forecast = float(np.mean(method_values)) if len(method_values) > 0 else 0.0

        # Stats
        avg_sales = float(np.mean(values))
        max_sales = float(np.max(values))
        min_sales = float(np.min(values))
        std_sales = float(np.std(values))
        trend_pct = float(((values[-1] - values[0]) / values[0]) * 100) if values[0] != 0 else 0.0
        
        # Classify trend
        trend_label = self.classify_trend_label(avg_sales, avg_forecast)

        # Prepare historical fields in both new and legacy shapes
        try:
            hist_values_list = [float(v) for v in values]
        except Exception:
            hist_values_list = list(values)

        if self.frequency == 'monthly':
            # represent historical_periods as strings like 'YYYY-MM'
            hist_periods_serial = [str(p) for p in periods]
            # For monthly, convert to year integers for the legacy field
            try:
                hist_years_list = [int(p.to_timestamp().year) for p in periods]
            except Exception:
                hist_years_list = [str(p) for p in periods]
        else:
            hist_periods_serial = [int(p) for p in periods]
            hist_years_list = [int(p) for p in periods]

        result = {
            'ref_article': ref_article,
            'designation': meta.get('designation'),
            'marque': meta.get('marque'),
            'famille': meta.get('famille'),
            'frequency': self.frequency,
            'next_period': next_period,
            # legacy key expected in many places
            'next_year': next_year,
            'avg_forecast': float(avg_forecast),
            'trend_label': trend_label,
            # Historical representations (new/serialized) for Streamlit
            'historical_periods': json.dumps(hist_periods_serial),
            'historical_values': json.dumps(hist_values_list),
            # Legacy list-shaped fields expected by API consumers
            'historical_years': hist_years_list,
            'historical_values_list': hist_values_list,
            'avg_sales': avg_sales,
            'max_sales': max_sales,
            'min_sales': min_sales,
            'std_sales': std_sales,
            'trend_pct': trend_pct,
            'data_points': len(values)
        }
        
        # Add individual forecasts
        for key, val in results.items():
            result[key] = float(val) if val is not None else np.nan
        
        # Add metrics as JSON strings (for CSV storage)
        for key, metrics in metrics_dict.items():
            if metrics:
                result[key] = json.dumps(metrics)
            else:
                result[key] = None

        # Save cache
        df_out = pd.DataFrame([result])
        self._save_forecast_cache(ref_article, df_out)

        return result

    # -------------------------
    # Forecast all articles
    # -------------------------
    def forecast_all_articles(self, period=3, alpha=0.3, force_recompute=False,
                              include_methods=None, fast_mode=True, progress_callback=None):
        if self.grouped_data is None:
            self.prepare_data()

        articles = sorted(self.grouped_data[self.ref_col].unique().tolist())
        all_results = []
        total = len(articles)
        for i, a in enumerate(articles, start=1):
            if progress_callback:
                try:
                    progress_callback(i, total)
                except Exception:
                    pass
            res = self.forecast_article(a, period=period, alpha=alpha, force_recompute=force_recompute,
                                        include_methods=include_methods, fast_mode=fast_mode)
            if res:
                all_results.append(res)

        df_all = pd.DataFrame(all_results)
        if not df_all.empty:
            df_all['ref_article'] = df_all['ref_article'].astype(str)
            df_all = df_all.sort_values('avg_forecast', ascending=False).reset_index(drop=True)
            
            # For parquet storage, drop list columns (they're stored as JSON strings anyway)
            list_cols = []
            for col in df_all.columns:
                try:
                    # Check if column contains lists
                    if df_all[col].dtype == 'object':
                        sample_val = df_all[col].dropna().iloc[0] if len(df_all[col].dropna()) > 0 else None
                        if isinstance(sample_val, list):
                            list_cols.append(col)
                except Exception:
                    pass
            
            if list_cols:
                df_all = df_all.drop(columns=list_cols)

        summary_path = self._summary_cache_path()
        df_all.to_parquet(summary_path, index=False)

        self.forecast_results = df_all
        return df_all

    # -------------------------
    # Utility methods
    # -------------------------
    def classify_trend_label(self, last_actual_mean, next_forecast_mean, tol=0.05):
        if last_actual_mean == 0:
            return "Stable"
        change = (next_forecast_mean - last_actual_mean) / last_actual_mean
        if change > tol:
            return "Uptrend"
        elif change < -tol:
            return "Downtrend"
        else:
            return "Stable"

    def generate_summary(self, lookback_periods=3, force_recompute=False):
        summary_path = self._summary_cache_path()
        if (not force_recompute) and summary_path.exists():
            try:
                df = pd.read_parquet(summary_path)
                self.forecast_results = df
                return df
            except Exception:
                pass

        if self.grouped_data is None:
            self.prepare_data()

        articles = sorted(self.grouped_data[self.ref_col].unique().tolist())
        recs = []
        for a in articles:
            f = self._load_forecast_cache(a, use_cache=True)
            if f is None:
                res = self.forecast_article(a, fast_mode=True)
                f = pd.DataFrame([res]) if res is not None else None
            if f is None or f.empty:
                continue

            row = f.iloc[0].to_dict()
            recs.append(row)

        df_summary = pd.DataFrame(recs)
        if not df_summary.empty:
            df_summary['ref_article'] = df_summary['ref_article'].astype(str)
            df_summary = df_summary.sort_values('avg_forecast', ascending=False).reset_index(drop=True)
            df_summary.to_parquet(summary_path, index=False)
        self.forecast_results = df_summary
        return df_summary

    def clear_model_cache(self):
        for p in self.model_cache.glob("*.pkl"):
            try:
                p.unlink()
            except Exception:
                pass

    def clear_forecast_cache(self):
        for p in self.forecast_cache.glob("*.csv"):
            try:
                p.unlink()
            except Exception:
                pass

    def clear_summary_cache(self):
        sp = self._summary_cache_path()
        if sp.exists():
            try:
                sp.unlink()
            except Exception:
                pass

    def clear_all_caches(self):
        self.clear_model_cache()
        self.clear_forecast_cache()
        self.clear_summary_cache()

