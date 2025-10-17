"""
Price action feature engineering for gold scalping.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any


class PriceActionFeatures:
    """Calculate price action-based features."""
    
    @staticmethod
    def calculate_candle_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic candle metrics.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with candle metrics
        """
        # Body size (absolute and ratio)
        df['body_size'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        
        # Avoid division by zero
        df['close_open_ratio'] = np.where(
            df['open'] != 0,
            (df['close'] - df['open']) / df['open'],
            0
        )
        
        # Upper and lower wicks
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['wick_size'] = df['upper_wick'] + df['lower_wick']
        
        # Wick ratios
        df['upper_wick_ratio'] = np.where(
            df['candle_range'] != 0,
            df['upper_wick'] / df['candle_range'],
            0
        )
        df['lower_wick_ratio'] = np.where(
            df['candle_range'] != 0,
            df['lower_wick'] / df['candle_range'],
            0
        )
        
        return df
    
    @staticmethod
    def detect_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect common candle patterns.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with pattern indicators
        """
        # Doji: small body relative to range
        body_to_range = np.where(
            df['candle_range'] != 0,
            df['body_size'] / df['candle_range'],
            0
        )
        df['is_doji'] = (body_to_range < 0.1).astype(int)
        
        # Bullish/Bearish candles
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        
        # Engulfing patterns
        prev_body = df['body_size'].shift(1)
        prev_bullish = df['is_bullish'].shift(1)
        
        df['is_bullish_engulfing'] = (
            (df['is_bullish'] == 1) &
            (prev_bullish == 0) &
            (df['body_size'] > prev_body * 1.5)
        ).astype(int)
        
        df['is_bearish_engulfing'] = (
            (df['is_bearish'] == 1) &
            (prev_bullish == 1) &
            (df['body_size'] > prev_body * 1.5)
        ).astype(int)
        
        df['is_engulfing'] = (df['is_bullish_engulfing'] | df['is_bearish_engulfing']).astype(int)
        
        # Pin bars (hammer/shooting star)
        df['is_pin_bar'] = (
            ((df['upper_wick_ratio'] > 0.6) | (df['lower_wick_ratio'] > 0.6)) &
            (body_to_range < 0.3)
        ).astype(int)
        
        df['is_hammer'] = (
            (df['lower_wick_ratio'] > 0.6) &
            (body_to_range < 0.3) &
            (df['upper_wick_ratio'] < 0.1)
        ).astype(int)
        
        df['is_shooting_star'] = (
            (df['upper_wick_ratio'] > 0.6) &
            (body_to_range < 0.3) &
            (df['lower_wick_ratio'] < 0.1)
        ).astype(int)
        
        return df
    
    @staticmethod
    def calculate_momentum(df: pd.DataFrame, periods: list = [3, 5, 10]) -> pd.DataFrame:
        """
        Calculate price momentum over different periods.
        
        Args:
            df: DataFrame with close prices
            periods: List of periods for momentum calculation
            
        Returns:
            DataFrame with momentum features
        """
        for period in periods:
            # Rate of change
            df[f'price_momentum_{period}'] = (
                (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
            ).fillna(0)
            
            # Directional momentum
            df[f'momentum_positive_{period}'] = (df[f'price_momentum_{period}'] > 0).astype(int)
        
        return df
    
    @staticmethod
    def calculate_price_position(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        Calculate price position relative to recent high/low.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Lookback period
            
        Returns:
            DataFrame with position features
        """
        # Rolling high/low
        df[f'high_{lookback}'] = df['high'].rolling(window=lookback).max()
        df[f'low_{lookback}'] = df['low'].rolling(window=lookback).min()
        
        # Distance to high/low
        df[f'distance_to_high_{lookback}'] = (
            (df[f'high_{lookback}'] - df['close']) / df['close']
        ).fillna(0)
        
        df[f'distance_to_low_{lookback}'] = (
            (df['close'] - df[f'low_{lookback}']) / df['close']
        ).fillna(0)
        
        # Position in range (0 = at low, 1 = at high)
        range_size = df[f'high_{lookback}'] - df[f'low_{lookback}']
        df[f'position_in_range_{lookback}'] = np.where(
            range_size != 0,
            (df['close'] - df[f'low_{lookback}']) / range_size,
            0.5
        )
        
        return df
    
    @staticmethod
    def calculate_volatility(df: pd.DataFrame, periods: list = [10, 20]) -> pd.DataFrame:
        """
        Calculate price volatility metrics.
        
        Args:
            df: DataFrame with close prices
            periods: List of periods for volatility calculation
            
        Returns:
            DataFrame with volatility features
        """
        for period in periods:
            # Standard deviation of returns
            returns = df['close'].pct_change()
            df[f'volatility_{period}'] = returns.rolling(window=period).std().fillna(0)
            
            # Average True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(window=period).mean().fillna(0)
        
        return df
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        Calculate all price action features.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Lookback period for calculations
            
        Returns:
            DataFrame with all price action features
        """
        df = PriceActionFeatures.calculate_candle_metrics(df)
        df = PriceActionFeatures.detect_candle_patterns(df)
        df = PriceActionFeatures.calculate_momentum(df, periods=[3, 5, 10])
        df = PriceActionFeatures.calculate_price_position(df, lookback=lookback)
        df = PriceActionFeatures.calculate_volatility(df, periods=[10, 20])
        
        return df
    
    @staticmethod
    def extract_features_for_ml(df: pd.DataFrame, index: int) -> Dict[str, Any]:
        """
        Extract price action features for a specific candle for ML model.
        
        Args:
            df: DataFrame with all features
            index: Index of the candle
            
        Returns:
            Dictionary of features
        """
        if index < 0 or index >= len(df):
            return {}
        
        features = {
            'body_size': df.loc[index, 'body_size'],
            'wick_size': df.loc[index, 'wick_size'],
            'candle_range': df.loc[index, 'candle_range'],
            'close_open_ratio': df.loc[index, 'close_open_ratio'],
            'is_doji': df.loc[index, 'is_doji'],
            'is_engulfing': df.loc[index, 'is_engulfing'],
            'is_pin_bar': df.loc[index, 'is_pin_bar'],
            'price_momentum_3': df.loc[index, 'price_momentum_3'],
            'price_momentum_5': df.loc[index, 'price_momentum_5'],
            'price_momentum_10': df.loc[index, 'price_momentum_10'],
            'distance_to_high_20': df.loc[index, 'distance_to_high_20'],
            'distance_to_low_20': df.loc[index, 'distance_to_low_20'],
        }
        
        return features
