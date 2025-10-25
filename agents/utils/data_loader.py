"""
Data Loader - Loads and indexes business data from CSV files.

This module provides efficient loading, caching, and querying of business data
from ventes_cleann.csv (sales transactions) and stock files (inventory).
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import json

logger = logging.getLogger("agents.data_loader")


class DataLoader:
    """
    Loads and manages business data from CSV files.
    
    Supports:
    - Sales data (ventes_cleann.csv) - 300k+ transaction rows
    - Stock data (STOCK.xlsx or stock_month.csv) - current inventory
    - Schema validation and normalization
    - In-memory caching with TTL
    - Efficient filtering and aggregation
    """
    
    def __init__(self, sales_path: Optional[str] = None, stock_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            sales_path: Path to sales CSV file (default: ventes_cleann.csv)
            stock_path: Path to stock file (default: STOCK.xlsx)
        """
        self.sales_path = Path(sales_path) if sales_path else Path("ventes_cleann.csv")
        self.stock_path = Path(stock_path) if stock_path else Path("STOCK.xlsx")
        
        self._sales_df: Optional[pd.DataFrame] = None
        self._stock_df: Optional[pd.DataFrame] = None
        self._cache: Dict[str, Any] = {}
        self._load_timestamp: Optional[datetime] = None
        
        # Expected schemas
        self.SALES_COLUMNS = [
            'Client_Principal', 'Code_Client', 'Intitule_Client', 'Categorie_Client',
            'Pays', 'Zone', 'Gouvernorat', 'Adresse', 'Representant',
            'NDocument', 'Type_Document', 'Date', 'Mois', 'Annee',
            'Marque', 'Famille', 'Sous_Famille', 'Ref_Article', 'Designation',
            'Qte_Vendu', 'CA_HT_BRUT', 'Tx_Remise', 'CA_HT_NET'
        ]
        
        self.STOCK_COLUMNS = [
            'Référence_Article', 'Désignation', 'Stock_à_terme'
        ]
        
        logger.info(f"DataLoader initialized: sales={self.sales_path}, stock={self.stock_path}")
    
    def load_sales(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load sales data from CSV.
        
        Args:
            force_reload: Force reload from disk even if cached
            
        Returns:
            DataFrame with sales data
        """
        if self._sales_df is not None and not force_reload:
            logger.debug("Using cached sales data")
            return self._sales_df
        
        logger.info(f"Loading sales data from {self.sales_path}")
        start = datetime.now()
        
        try:
            # Read with semicolon separator and latin1 encoding
            df = pd.read_csv(
                self.sales_path,
                sep=';',
                encoding='latin1',
                low_memory=False
            )
            
            # Clean BOM from column names
            df.columns = df.columns.str.replace('\ufeff', '').str.strip()
            
            # Normalize column names (remove accents, spaces)
            column_mapping = {}
            for col in df.columns:
                clean_col = col.replace('é', 'e').replace('è', 'e').replace('ï', 'i')
                clean_col = clean_col.replace('Intitule_client', 'Intitule_Client')
                column_mapping[col] = clean_col
            
            df.rename(columns=column_mapping, inplace=True)
            
            # Convert data types
            # Parse dates using day-first (CSV uses dd/mm/YYYY) and try to infer format
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce', infer_datetime_format=True)

            # Ensure key identifiers remain strings and normalize formatting
            if 'Ref_Article' in df.columns:
                # Some exports use comma in scientific notation or thousands separators
                # Keep as string and strip commas/whitespace to avoid float coercion
                df['Ref_Article'] = df['Ref_Article'].astype(str).str.replace(',', '').str.strip()

            if 'Code_Client' in df.columns:
                df['Code_Client'] = df['Code_Client'].astype(str).str.strip()
            
            numeric_cols = ['Qte_Vendu', 'CA_HT_BRUT', 'Tx_Remise', 'CA_HT_NET', 'Annee', 'Mois']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove duplicate Annee column if exists
            if 'Annee' in df.columns and 'AnnÃ©e' in df.columns:
                df.drop('AnnÃ©e', axis=1, inplace=True, errors='ignore')
            
            # Cache the dataframe
            self._sales_df = df
            self._load_timestamp = datetime.now()
            
            duration = (datetime.now() - start).total_seconds()
            logger.info(f"Loaded {len(df)} sales records in {duration:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load sales data: {e}")
            raise
    
    def load_stock(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load stock data from Excel or CSV.
        
        Args:
            force_reload: Force reload from disk even if cached
            
        Returns:
            DataFrame with stock data
        """
        if self._stock_df is not None and not force_reload:
            logger.debug("Using cached stock data")
            return self._stock_df
        
        logger.info(f"Loading stock data from {self.stock_path}")
        start = datetime.now()
        
        try:
            # Try Excel first, then CSV
            if self.stock_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(self.stock_path)
            else:
                df = pd.read_csv(self.stock_path, sep='\t', encoding='utf-8')
            
            # Normalize column names
            df.columns = df.columns.str.strip()
            
            # Standardize column names
            column_mapping = {
                'Référence Article': 'Référence_Article',
                'Reference Article': 'Référence_Article',
                'Ref_Article': 'Référence_Article',
                'Stock à terme': 'Stock_à_terme',
                'Stock a terme': 'Stock_à_terme',
                'Désignation': 'Désignation',
                'Designation': 'Désignation'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
            
            # Convert stock to numeric
            if 'Stock_à_terme' in df.columns:
                df['Stock_à_terme'] = pd.to_numeric(df['Stock_à_terme'], errors='coerce')
            
            # Cache the dataframe
            self._stock_df = df
            
            duration = (datetime.now() - start).total_seconds()
            logger.info(f"Loaded {len(df)} stock records in {duration:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load stock data: {e}")
            raise
    
    def get_sales_data(self) -> pd.DataFrame:
        """Get sales data (loads if not already loaded)."""
        if self._sales_df is None:
            return self.load_sales()
        return self._sales_df
    
    def get_stock_data(self) -> pd.DataFrame:
        """Get stock data (loads if not already loaded)."""
        if self._stock_df is None:
            return self.load_stock()
        return self._stock_df
    
    def get_client_info(self, code_client: str) -> Optional[Dict[str, Any]]:
        """
        Get client information by code.
        
        Args:
            code_client: Client code (e.g., 'CL0001')
            
        Returns:
            Dictionary with client details or None if not found
        """
        df = self.get_sales_data()
        
        client_data = df[df['Code_Client'] == code_client]
        
        if client_data.empty:
            return None
        
        # Get most recent record
        latest = client_data.iloc[0]
        
        return {
            'code_client': code_client,
            'intitule': latest.get('Intitule_Client'),
            'categorie': latest.get('Categorie_Client'),
            'pays': latest.get('Pays'),
            'zone': latest.get('Zone'),
            'gouvernorat': latest.get('Gouvernorat'),
            'representant': latest.get('Representant'),
            'total_transactions': len(client_data),
            'total_ca': float(client_data['CA_HT_NET'].sum()) if 'CA_HT_NET' in client_data else 0
        }
    
    def get_product_info(self, ref_article: str) -> Optional[Dict[str, Any]]:
        """
        Get product information by reference.
        
        Args:
            ref_article: Product reference
            
        Returns:
            Dictionary with product details or None if not found
        """
        sales_df = self.get_sales_data()
        stock_df = self.get_stock_data()
        
        # Get from sales
        product_sales = sales_df[sales_df['Ref_Article'] == ref_article]
        
        if product_sales.empty:
            return None
        
        latest = product_sales.iloc[0]
        
        # Get stock if available
        stock_qty = None
        if 'Référence_Article' in stock_df.columns:
            stock_row = stock_df[stock_df['Référence_Article'] == ref_article]
            if not stock_row.empty:
                stock_qty = float(stock_row.iloc[0]['Stock_à_terme'])
        
        return {
            'ref_article': ref_article,
            'designation': latest.get('Designation'),
            'marque': latest.get('Marque'),
            'famille': latest.get('Famille'),
            'sous_famille': latest.get('Sous_Famille'),
            'stock_actuel': stock_qty,
            'total_ventes': float(product_sales['Qte_Vendu'].sum()) if 'Qte_Vendu' in product_sales else 0,
            'total_ca': float(product_sales['CA_HT_NET'].sum()) if 'CA_HT_NET' in product_sales else 0
        }
    
    def query_sales(
        self,
        filters: Optional[Dict[str, Any]] = None,
        group_by: Optional[List[str]] = None,
        aggregations: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Query sales data with filters and aggregations.
        
        Args:
            filters: Dictionary of column: value filters
            group_by: Columns to group by
            aggregations: Dictionary of column: aggregation_function
            
        Returns:
            Filtered and aggregated DataFrame
        """
        df = self.get_sales_data().copy()
        
        # Apply filters
        if filters:
            for col, value in filters.items():
                if col in df.columns:
                    if isinstance(value, list):
                        df = df[df[col].isin(value)]
                    else:
                        df = df[df[col] == value]
        
        # Apply grouping and aggregation
        if group_by and aggregations:
            df = df.groupby(group_by).agg(aggregations).reset_index()
        
        return df
    
    def get_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """Generate cache key for operation."""
        key_str = f"{operation}_{json.dumps(params, sort_keys=True, default=str)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(cache_key)
    
    def set_cache(self, cache_key: str, value: Any, ttl_seconds: int = 300):
        """Set value in cache with TTL."""
        self._cache[cache_key] = {
            'value': value,
            'expires_at': datetime.now().timestamp() + ttl_seconds
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            'sales_loaded': self._sales_df is not None,
            'stock_loaded': self._stock_df is not None,
            'sales_rows': len(self._sales_df) if self._sales_df is not None else 0,
            'stock_rows': len(self._stock_df) if self._stock_df is not None else 0,
            'load_timestamp': self._load_timestamp.isoformat() if self._load_timestamp else None,
            'cache_size': len(self._cache)
        }


# Global singleton instance
_data_loader: Optional[DataLoader] = None


def get_data_loader() -> DataLoader:
    """Get or create global DataLoader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader
