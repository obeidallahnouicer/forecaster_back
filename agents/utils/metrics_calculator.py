"""
Business Metrics Calculator - Computes KPIs from business data.

Calculates key performance indicators:
- ΔCA_%: Change in revenue percentage
- ΔQte_%: Change in quantity percentage
- Couverture_Stock: Stock coverage in months
- Rotation_Mensuelle: Monthly rotation rate
- Tendance_3m: 3-month trend
- Growth metrics and classifications
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger("agents.metrics_calculator")


@dataclass
class ProductMetrics:
    """Computed metrics for a product."""
    ref_article: str
    designation: Optional[str]
    
    # Sales metrics
    ca_actuel: float
    ca_precedent: float
    delta_ca_pct: float
    
    qte_actuelle: float
    qte_precedente: float
    delta_qte_pct: float
    
    # Stock metrics
    stock_actuel: Optional[float]
    rotation_mensuelle: Optional[float]
    couverture_stock: Optional[float]
    
    # Trend metrics
    tendance_3m: Optional[str]  # 'hausse', 'baisse', 'stable'
    tendance_pct: Optional[float]
    
    # Classification
    statut_stock: Optional[str]  # 'rupture', 'normal', 'surstock'
    statut_ventes: Optional[str]  # 'croissance', 'stable', 'declin'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ref_article': self.ref_article,
            'designation': self.designation,
            'ca_actuel': self.ca_actuel,
            'ca_precedent': self.ca_precedent,
            'delta_ca_pct': self.delta_ca_pct,
            'qte_actuelle': self.qte_actuelle,
            'qte_precedente': self.qte_precedente,
            'delta_qte_pct': self.delta_qte_pct,
            'stock_actuel': self.stock_actuel,
            'rotation_mensuelle': self.rotation_mensuelle,
            'couverture_stock': self.couverture_stock,
            'tendance_3m': self.tendance_3m,
            'tendance_pct': self.tendance_pct,
            'statut_stock': self.statut_stock,
            'statut_ventes': self.statut_ventes
        }


class MetricsCalculator:
    """
    Calculates business KPIs from sales and stock data.
    
    Supports:
    - Period-over-period comparisons (month, quarter, year)
    - Stock coverage analysis
    - Trend detection (3-month rolling)
    - Growth rate calculations
    - Classification (risk, opportunity, stable)
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        logger.info("MetricsCalculator initialized")
    
    def calculate_product_metrics(
        self,
        sales_df: pd.DataFrame,
        stock_df: Optional[pd.DataFrame],
        ref_article: str,
        current_period: Optional[str] = None,
        previous_period: Optional[str] = None
    ) -> ProductMetrics:
        """
        Calculate all metrics for a single product.
        
        Args:
            sales_df: Sales DataFrame
            stock_df: Stock DataFrame (optional)
            ref_article: Product reference
            current_period: Current period filter (e.g., '2024-09')
            previous_period: Previous period filter (e.g., '2024-08')
            
        Returns:
            ProductMetrics with all calculated KPIs
        """
        # Filter sales for this product
        product_sales = sales_df[sales_df['Ref_Article'] == ref_article].copy()
        
        if product_sales.empty:
            raise ValueError(f"No sales data found for {ref_article}")
        
        # Get designation
        designation = product_sales.iloc[0].get('Designation')
        
        # Calculate current period metrics
        if current_period:
            current_data = product_sales[product_sales['Date'].dt.strftime('%Y-%m') == current_period]
        else:
            # Use last month
            max_date = product_sales['Date'].max()
            current_data = product_sales[product_sales['Date'].dt.to_period('M') == max_date.to_period('M')]
        
        ca_actuel = float(current_data['CA_HT_NET'].sum()) if not current_data.empty else 0.0
        qte_actuelle = float(current_data['Qte_Vendu'].sum()) if not current_data.empty else 0.0
        
        # Calculate previous period metrics
        if previous_period:
            previous_data = product_sales[product_sales['Date'].dt.strftime('%Y-%m') == previous_period]
        else:
            # Use month before current
            if current_period:
                prev_date = pd.to_datetime(current_period) - pd.DateOffset(months=1)
                previous_data = product_sales[product_sales['Date'].dt.to_period('M') == prev_date.to_period('M')]
            else:
                max_date = product_sales['Date'].max()
                prev_month = (max_date - pd.DateOffset(months=1)).to_period('M')
                previous_data = product_sales[product_sales['Date'].dt.to_period('M') == prev_month]
        
        ca_precedent = float(previous_data['CA_HT_NET'].sum()) if not previous_data.empty else 0.0
        qte_precedente = float(previous_data['Qte_Vendu'].sum()) if not previous_data.empty else 0.0
        
        # Calculate deltas
        delta_ca_pct = self._calculate_percentage_change(ca_actuel, ca_precedent)
        delta_qte_pct = self._calculate_percentage_change(qte_actuelle, qte_precedente)
        
        # Calculate stock metrics
        stock_actuel = None
        rotation_mensuelle = None
        couverture_stock = None
        
        if stock_df is not None and 'Référence_Article' in stock_df.columns:
            stock_row = stock_df[stock_df['Référence_Article'] == ref_article]
            if not stock_row.empty:
                stock_actuel = float(stock_row.iloc[0]['Stock_à_terme'])
                
                # Calculate monthly rotation (average monthly sales)
                rotation_mensuelle = self._calculate_monthly_rotation(product_sales)
                
                # Calculate stock coverage
                if rotation_mensuelle and rotation_mensuelle > 0:
                    couverture_stock = stock_actuel / rotation_mensuelle
                else:
                    couverture_stock = None
        
        # Calculate 3-month trend
        tendance_3m, tendance_pct = self._calculate_trend_3m(product_sales)
        
        # Classify stock status
        statut_stock = self._classify_stock_status(couverture_stock)
        
        # Classify sales status
        statut_ventes = self._classify_sales_status(delta_ca_pct, delta_qte_pct)
        
        return ProductMetrics(
            ref_article=ref_article,
            designation=designation,
            ca_actuel=ca_actuel,
            ca_precedent=ca_precedent,
            delta_ca_pct=delta_ca_pct,
            qte_actuelle=qte_actuelle,
            qte_precedente=qte_precedente,
            delta_qte_pct=delta_qte_pct,
            stock_actuel=stock_actuel,
            rotation_mensuelle=rotation_mensuelle,
            couverture_stock=couverture_stock,
            tendance_3m=tendance_3m,
            tendance_pct=tendance_pct,
            statut_stock=statut_stock,
            statut_ventes=statut_ventes
        )
    
    def calculate_client_metrics(
        self,
        sales_df: pd.DataFrame,
        code_client: str,
        months_lookback: int = 3
    ) -> Dict[str, Any]:
        """
        Calculate metrics for a client.
        
        Args:
            sales_df: Sales DataFrame
            code_client: Client code
            months_lookback: Number of months to look back for activity check
            
        Returns:
            Dictionary with client metrics
        """
        client_sales = sales_df[sales_df['Code_Client'] == code_client].copy()
        
        if client_sales.empty:
            return {
                'code_client': code_client,
                'actif': False,
                'derniere_transaction': None,
                'ca_total': 0.0,
                'nb_transactions': 0
            }
        
        # Sort by date
        client_sales = client_sales.sort_values('Date', ascending=False)
        
        # Check if active in last N months
        max_date = sales_df['Date'].max()
        cutoff_date = max_date - pd.DateOffset(months=months_lookback)
        recent_sales = client_sales[client_sales['Date'] >= cutoff_date]
        
        actif = len(recent_sales) > 0
        derniere_transaction = client_sales.iloc[0]['Date'] if not client_sales.empty else None
        
        # Calculate totals
        ca_total = float(client_sales['CA_HT_NET'].sum())
        nb_transactions = len(client_sales)
        
        # Calculate average basket
        ca_moyen = ca_total / nb_transactions if nb_transactions > 0 else 0.0
        
        return {
            'code_client': code_client,
            'actif': actif,
            'derniere_transaction': derniere_transaction.strftime('%Y-%m-%d') if derniere_transaction else None,
            'ca_total': ca_total,
            'nb_transactions': nb_transactions,
            'ca_moyen': ca_moyen,
            'mois_depuis_derniere_vente': self._months_since(derniere_transaction, max_date) if derniere_transaction else None
        }
    
    def _calculate_percentage_change(self, current: float, previous: float) -> float:
        """Calculate percentage change."""
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        return ((current - previous) / previous) * 100.0
    
    def _calculate_monthly_rotation(self, sales_df: pd.DataFrame) -> Optional[float]:
        """Calculate average monthly sales quantity."""
        if sales_df.empty or 'Date' not in sales_df.columns:
            return None
        
        # Group by month
        sales_df = sales_df.copy()
        sales_df['YearMonth'] = sales_df['Date'].dt.to_period('M')
        monthly_sales = sales_df.groupby('YearMonth')['Qte_Vendu'].sum()
        
        if len(monthly_sales) == 0:
            return None
        
        # Average over available months
        return float(monthly_sales.mean())
    
    def _calculate_trend_3m(self, sales_df: pd.DataFrame) -> tuple:
        """Calculate 3-month trend."""
        if sales_df.empty or len(sales_df) < 2:
            return 'stable', 0.0
        
        # Get last 3 months
        sales_df = sales_df.copy()
        sales_df = sales_df.sort_values('Date')
        max_date = sales_df['Date'].max()
        cutoff_date = max_date - pd.DateOffset(months=3)
        recent = sales_df[sales_df['Date'] >= cutoff_date]
        
        if recent.empty:
            return 'stable', 0.0
        
        # Group by month and calculate CA
        recent['YearMonth'] = recent['Date'].dt.to_period('M')
        monthly_ca = recent.groupby('YearMonth')['CA_HT_NET'].sum().reset_index()
        monthly_ca = monthly_ca.sort_values('YearMonth')
        
        if len(monthly_ca) < 2:
            return 'stable', 0.0
        
        # Calculate trend using linear regression slope
        x = np.arange(len(monthly_ca))
        y = monthly_ca['CA_HT_NET'].values
        
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            avg_ca = y.mean()
            
            if avg_ca > 0:
                trend_pct = (slope / avg_ca) * 100.0
            else:
                trend_pct = 0.0
            
            # Classify
            if trend_pct > 5:
                return 'hausse', trend_pct
            elif trend_pct < -5:
                return 'baisse', trend_pct
            else:
                return 'stable', trend_pct
        
        return 'stable', 0.0
    
    def _classify_stock_status(self, couverture: Optional[float]) -> Optional[str]:
        """Classify stock status based on coverage."""
        if couverture is None:
            return None
        
        if couverture < 1.0:
            return 'rupture'
        elif couverture > 6.0:
            return 'surstock'
        else:
            return 'normal'
    
    def _classify_sales_status(self, delta_ca_pct: float, delta_qte_pct: float) -> str:
        """Classify sales status based on deltas."""
        avg_delta = (delta_ca_pct + delta_qte_pct) / 2.0
        
        if avg_delta > 10.0:
            return 'croissance'
        elif avg_delta < -10.0:
            return 'declin'
        else:
            return 'stable'
    
    def _months_since(self, date: datetime, reference_date: datetime) -> int:
        """Calculate months between two dates."""
        if pd.isna(date) or pd.isna(reference_date):
            return 0
        
        delta = (reference_date.year - date.year) * 12 + (reference_date.month - date.month)
        return max(0, delta)


# Global singleton
_metrics_calculator: Optional[MetricsCalculator] = None


def get_metrics_calculator() -> MetricsCalculator:
    """Get or create global MetricsCalculator instance."""
    global _metrics_calculator
    if _metrics_calculator is None:
        _metrics_calculator = MetricsCalculator()
    return _metrics_calculator
