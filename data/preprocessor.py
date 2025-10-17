"""
Data preprocessing and validation for the trading system.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from utils.helpers import validate_ohlc
from utils.logger import TradingLogger


class DataPreprocessor:
    """Handles data cleaning, validation, and preprocessing."""
    
    def __init__(self, logger: Optional[TradingLogger] = None):
        """
        Initialize data preprocessor.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or TradingLogger.get_logger(__name__)
    
    def clean_candle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate candle data.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Cleaning {len(df)} candles")
        
        # Validate OHLC integrity
        df = validate_ohlc(df)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        self.logger.info(f"Cleaned data: {len(df)} candles remaining")
        return df
    
    def fill_missing_candles(self, df: pd.DataFrame, interval_minutes: int = 5) -> pd.DataFrame:
        """
        Fill missing candles with forward-filled data.
        
        Args:
            df: DataFrame with candle data
            interval_minutes: Candle interval in minutes
            
        Returns:
            DataFrame with filled gaps
        """
        if len(df) == 0:
            return df
        
        # Create complete timestamp range
        start_ts = df['timestamp'].min()
        end_ts = df['timestamp'].max()
        
        complete_range = pd.date_range(
            start=pd.to_datetime(start_ts, unit='s'),
            end=pd.to_datetime(end_ts, unit='s'),
            freq=f'{interval_minutes}T'
        )
        
        complete_df = pd.DataFrame({
            'timestamp': complete_range.astype(np.int64) // 10**9
        })
        
        # Merge with existing data
        merged = complete_df.merge(df, on='timestamp', how='left')
        
        # Forward fill missing values
        merged[['open', 'high', 'low', 'close']] = merged[['open', 'high', 'low', 'close']].fillna(method='ffill')
        merged['volume'] = merged['volume'].fillna(0)
        
        gaps_filled = len(merged) - len(df)
        if gaps_filled > 0:
            self.logger.warning(f"Filled {gaps_filled} missing candles")
        
        return merged
    
    def detect_outliers(self, df: pd.DataFrame, column: str = 'close',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and remove outliers using z-score method.
        
        Args:
            df: DataFrame with data
            column: Column to check for outliers
            threshold: Z-score threshold
            
        Returns:
            DataFrame with outliers removed
        """
        if len(df) < 10:
            return df
        
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold
        
        outlier_count = outliers.sum()
        if outlier_count > 0:
            self.logger.warning(f"Detected {outlier_count} outliers in {column}")
            df = df[~outliers]
        
        return df
    
    def add_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add datetime-based features to DataFrame.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with datetime features
        """
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def resample_timeframe(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample candle data to different timeframe.
        
        Args:
            df: DataFrame with OHLC data
            target_timeframe: Target timeframe (e.g., '15T', '1H')
            
        Returns:
            Resampled DataFrame
        """
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('datetime')
        
        resampled = df.resample(target_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        resampled['timestamp'] = resampled.index.astype(np.int64) // 10**9
        resampled = resampled.reset_index(drop=True)
        
        return resampled
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price returns.
        
        Args:
            df: DataFrame with close prices
            
        Returns:
            DataFrame with returns
        """
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, columns: list,
                          method: str = 'minmax') -> Tuple[pd.DataFrame, dict]:
        """
        Normalize feature columns.
        
        Args:
            df: DataFrame with features
            columns: Columns to normalize
            method: Normalization method ('minmax' or 'zscore')
            
        Returns:
            Tuple of (normalized DataFrame, normalization parameters)
        """
        params = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                
                if max_val - min_val > 0:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
            
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
                    params[col] = {'mean': mean_val, 'std': std_val, 'method': 'zscore'}
        
        return df, params
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: DataFrame with potential missing values
            strategy: Strategy to handle missing values ('forward_fill', 'backward_fill', 'drop', 'zero')
            
        Returns:
            DataFrame with handled missing values
        """
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            self.logger.warning(f"Handling {missing_count} missing values using {strategy}")
            
            if strategy == 'forward_fill':
                df = df.fillna(method='ffill')
            elif strategy == 'backward_fill':
                df = df.fillna(method='bfill')
            elif strategy == 'drop':
                df = df.dropna()
            elif strategy == 'zero':
                df = df.fillna(0)
        
        return df
