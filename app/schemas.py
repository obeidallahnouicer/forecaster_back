from pydantic import BaseModel
from typing import List, Optional, Any


class UploadResponse(BaseModel):
    session_id: str
    rows: int


class ArticleListResponse(BaseModel):
    count: int
    articles: List[str]


class ForecastArticleRequest(BaseModel):
    ref: str
    period: int = 3
    alpha: float = 0.3
    force_recompute: bool = False
    fast_mode: bool = True


class ForecastArticleResponse(BaseModel):
    ref_article: str
    designation: Optional[str]
    marque: Optional[str]
    famille: Optional[str]
    next_year: int
    sma_forecast: Optional[float]
    es_forecast: Optional[float]
    lr_forecast: Optional[float]
    arima_forecast: Optional[float]
    prophet_forecast: Optional[float]
    xgb_forecast: Optional[float]
    avg_forecast: float
    historical_years: List[Any]
    historical_values: List[float]
    avg_sales: float
    max_sales: float
    min_sales: float
    std_sales: float
    trend_pct: float
    data_points: int


class SummaryResponse(BaseModel):
    count: int
    rows: List[dict]


class GenericResponse(BaseModel):
    detail: str
