from pydantic import BaseModel
from typing import List, Optional, Any, Dict


class UploadResponse(BaseModel):
    session_id: str
    rows: int


class ArticleListResponse(BaseModel):
    count: int
    articles: List[str]


# ================== Forecast Request Models ==================

class ForecastArticleRequest(BaseModel):
    """Simple forecast request (minimal parameters)"""
    ref: str
    period: int = 3
    alpha: float = 0.3
    force_recompute: bool = False
    fast_mode: bool = True


class ForecastArticleRequestFull(BaseModel):
    """Full forecast request with all parameter choices"""
    ref: str
    period: int = 3
    alpha: float = 0.3
    force_recompute: bool = False
    fast_mode: bool = True
    include_methods: Optional[List[str]] = None
    # include_methods: Can be ['SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST']
    # or any subset. If None, all methods are used.


class ForecastAllRequest(BaseModel):
    """Request to forecast all articles with parameter choices"""
    period: int = 3
    alpha: float = 0.3
    force_recompute: bool = False
    fast_mode: bool = True
    include_methods: Optional[List[str]] = None


class MonthlyForecastRequest(BaseModel):
    """Request for monthly forecast with full options"""
    ref: str
    period: int = 3
    alpha: float = 0.3
    force_recompute: bool = False
    fast_mode: bool = True
    include_methods: Optional[List[str]] = None
    # Additional monthly options
    aggregate_by: str = "month"  # 'month', 'quarter', 'year'
    start_period: Optional[str] = None  # e.g., '2024-01' for monthly
    end_period: Optional[str] = None


# ================== Forecast Response Models ==================

class ForecastArticleResponse(BaseModel):
    ref_article: str
    designation: Optional[str]
    marque: Optional[str]
    famille: Optional[str]
    frequency: str
    next_period: Any
    next_year: Optional[int]
    sma_forecast: Optional[float]
    es_forecast: Optional[float]
    lr_forecast: Optional[float]
    arima_forecast: Optional[float]
    prophet_forecast: Optional[float]
    xgb_forecast: Optional[float]
    avg_forecast: float
    trend_label: str
    trend_pct: float
    # JSON serialized fields for Streamlit compatibility
    historical_periods: str
    historical_values: str
    # Legacy list fields
    historical_years: List[Any]
    historical_values_list: List[float]
    avg_sales: float
    max_sales: float
    min_sales: float
    std_sales: float
    data_points: int
    # Metrics as JSON strings
    sma_metrics: Optional[str]
    es_metrics: Optional[str]
    lr_metrics: Optional[str]
    arima_metrics: Optional[str]
    prophet_metrics: Optional[str]
    xgb_metrics: Optional[str]


class SummaryResponse(BaseModel):
    count: int
    rows: List[dict]


class GenericResponse(BaseModel):
    detail: str


# ================== Monthly-Specific Response Models ==================

class MonthlyPeriodInfo(BaseModel):
    """Information about a monthly period"""
    period: str  # e.g., '2024-01'
    year: int
    month: int
    total_sales: float
    article_count: int


class MonthlyArticleBreakdown(BaseModel):
    """Monthly breakdown for a single article"""
    ref_article: str
    designation: Optional[str]
    periods: List[MonthlyPeriodInfo]
    total_sales: float
    avg_monthly_sales: float
    trend_direction: str  # 'uptrend', 'downtrend', 'stable'


class MonthlyAggregateResponse(BaseModel):
    """Aggregated metrics across all months"""
    session_id: str
    frequency: str  # 'monthly'
    total_articles: int
    total_periods: int
    date_range_start: str
    date_range_end: str
    total_sales: float
    avg_monthly_sales: float
    summary: Dict[str, Any]  # Additional aggregated stats


class MonthlyForecastResponse(BaseModel):
    """Monthly forecast result"""
    ref_article: str
    designation: Optional[str]
    frequency: str  # 'monthly'
    next_month: str  # e.g., '2024-02'
    next_month_forecast: float
    monthly_history: List[Dict[str, Any]]  # Historical monthly data
    trend_label: str
    trend_pct: float
    method_forecasts: Dict[str, float]  # Per-method forecasts
    avg_forecast: float


# ================== Dashboard API schemas ==================

class DashboardDocument(BaseModel):
    """Single forecast row for dashboard."""
    ref_article: str
    designation: str
    marque: str
    famille: str
    next_year: int
    avg_forecast: float
    trend_pct: float
    trend_label: str
    data_points: int


class DocumentsResponse(BaseModel):
    """Response for GET /documents."""
    total_docs: int
    filtered_docs: int
    limit: int
    offset: int
    documents: List[DashboardDocument]


class StatusResponse(BaseModel):
    """Response for GET /status."""
    csv_exists: bool
    csv_path: str
    total_rows: int
    columns: List[str]


class ArticleMetric(BaseModel):
    """Article with a single metric."""
    ref_article: str
    designation: str
    metric_value: float


class TopArticle(BaseModel):
    """Top article by avg_forecast."""
    ref_article: str
    designation: str
    avg_forecast: float


class TopTrendArticle(BaseModel):
    """Top article by trend_pct."""
    ref_article: str
    designation: str
    trend_pct: float


class FamilleAggregation(BaseModel):
    """Aggregated metrics by famille."""
    famille: str
    total_avg_forecast: float
    count: int


class MarqueAggregation(BaseModel):
    """Aggregated metrics by marque."""
    marque: str
    total_avg_forecast: float
    count: int


class MetricsResponse(BaseModel):
    """Response for GET /metrics."""
    total_rows: int
    total_avg_forecast: float
    avg_trend_pct: float
    avg_data_points: float
    trend_counts: Dict[str, int]
    top_articles: List[TopArticle]
    top_marques: List[MarqueAggregation]
    top_familles: List[FamilleAggregation]
    products_by_famille: Dict[str, List[str]]
    products_by_marque: Dict[str, List[str]]
    avg_forecast_by_famille: Dict[str, float]
    avg_forecast_by_marque: Dict[str, float]
    top_trend_pct_articles: List[TopTrendArticle]

