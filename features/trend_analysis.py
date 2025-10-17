"""
Trend analysis and multi-timeframe trend detection.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class TrendAnalyzer:
    """Analyze price trends across multiple timeframes."""
    
    @staticmethod
    def detect_higher_highs_lows(df: pd.DataFrame, lookback: int = 20) -> Tuple[int, int]:
        """
        Count higher highs and higher lows.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of candles to analyze
            
        Returns:
            Tuple of (higher_highs_count, higher_lows_count)
        """
        if len(df) < lookback + 1:
            return 0, 0
        
        recent_df = df.tail(lookback + 1)
        
        higher_highs = 0
        higher_lows = 0
        
        for i in range(1, len(recent_df)):
            if recent_df.iloc[i]['high'] > recent_df.iloc[i-1]['high']:
                higher_highs += 1
            if recent_df.iloc[i]['low'] > recent_df.iloc[i-1]['low']:
                higher_lows += 1
        
        return higher_highs, higher_lows
    
    @staticmethod
    def detect_lower_highs_lows(df: pd.DataFrame, lookback: int = 20) -> Tuple[int, int]:
        """
        Count lower highs and lower lows.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of candles to analyze
            
        Returns:
            Tuple of (lower_highs_count, lower_lows_count)
        """
        if len(df) < lookback + 1:
            return 0, 0
        
        recent_df = df.tail(lookback + 1)
        
        lower_highs = 0
        lower_lows = 0
        
        for i in range(1, len(recent_df)):
            if recent_df.iloc[i]['high'] < recent_df.iloc[i-1]['high']:
                lower_highs += 1
            if recent_df.iloc[i]['low'] < recent_df.iloc[i-1]['low']:
                lower_lows += 1
        
        return lower_highs, lower_lows
    
    @staticmethod
    def calculate_trend_strength(df: pd.DataFrame, lookback: int = 20) -> float:
        """
        Calculate trend strength score (0-1).
        
        Args:
            df: DataFrame with close prices
            lookback: Number of candles to analyze
            
        Returns:
            Trend strength score
        """
        if len(df) < lookback:
            return 0.0
        
        recent_closes = df['close'].tail(lookback).values
        
        # Linear regression slope
        x = np.arange(len(recent_closes))
        slope, _ = np.polyfit(x, recent_closes, 1)
        
        # Normalize slope relative to price
        normalized_slope = abs(slope) / recent_closes.mean()
        
        # R-squared for trend consistency
        y_pred = slope * x + recent_closes[0]
        ss_res = np.sum((recent_closes - y_pred) ** 2)
        ss_tot = np.sum((recent_closes - recent_closes.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Combine slope and R-squared
        strength = min(normalized_slope * 100, 1.0) * r_squared
        
        return max(0.0, min(1.0, strength))
    
    @staticmethod
    def determine_trend_direction(df: pd.DataFrame, lookback: int = 20) -> int:
        """
        Determine trend direction.
        
        Args:
            df: DataFrame with close prices
            lookback: Number of candles to analyze
            
        Returns:
            1 for uptrend, -1 for downtrend, 0 for sideways
        """
        if len(df) < lookback:
            return 0
        
        recent_closes = df['close'].tail(lookback).values
        
        # Simple moving average comparison
        sma_short = recent_closes[-5:].mean()
        sma_long = recent_closes.mean()
        
        # Price position
        current_price = recent_closes[-1]
        
        # Determine direction
        if current_price > sma_long and sma_short > sma_long:
            return 1  # Uptrend
        elif current_price < sma_long and sma_short < sma_long:
            return -1  # Downtrend
        else:
            return 0  # Sideways
    
    @staticmethod
    def analyze_single_timeframe(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
        """
        Analyze trend for a single timeframe.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of candles to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        hh_count, hl_count = TrendAnalyzer.detect_higher_highs_lows(df, lookback)
        lh_count, ll_count = TrendAnalyzer.detect_lower_highs_lows(df, lookback)
        
        trend_strength = TrendAnalyzer.calculate_trend_strength(df, lookback)
        trend_direction = TrendAnalyzer.determine_trend_direction(df, lookback)
        
        return {
            'higher_highs_count': hh_count,
            'higher_lows_count': hl_count,
            'lower_highs_count': lh_count,
            'lower_lows_count': ll_count,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction
        }
    
    @staticmethod
    def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to different timeframe.
        
        Args:
            df: DataFrame with OHLC data and timestamp
            timeframe: Target timeframe ('15T', '1H', etc.)
            
        Returns:
            Resampled DataFrame
        """
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='s')
        df_copy = df_copy.set_index('datetime')
        
        resampled = df_copy.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        resampled['timestamp'] = resampled.index.astype(np.int64) // 10**9
        resampled = resampled.reset_index(drop=True)
        
        return resampled
    
    @staticmethod
    def multi_timeframe_analysis(df: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze trends across multiple timeframes.
        
        Args:
            df: DataFrame with 5-minute OHLC data
            
        Returns:
            Dictionary with trend directions for each timeframe
        """
        # 5-minute trend (current timeframe)
        trend_5m = TrendAnalyzer.determine_trend_direction(df, lookback=20)
        
        # 15-minute trend
        try:
            df_15m = TrendAnalyzer.resample_to_timeframe(df, '15min')
            trend_15m = TrendAnalyzer.determine_trend_direction(df_15m, lookback=20)
        except:
            trend_15m = 0
        
        # 1-hour trend
        try:
            df_1h = TrendAnalyzer.resample_to_timeframe(df, '1h')
            trend_1h = TrendAnalyzer.determine_trend_direction(df_1h, lookback=20)
        except:
            trend_1h = 0
        
        return {
            'trend_5m': trend_5m,
            'trend_15m': trend_15m,
            'trend_1h': trend_1h
        }
    
    @staticmethod
    def check_trend_alignment(mtf_trends: Dict[str, int]) -> bool:
        """
        Check if trends are aligned across timeframes.
        
        Args:
            mtf_trends: Multi-timeframe trend dictionary
            
        Returns:
            True if trends are aligned
        """
        trends = [mtf_trends['trend_5m'], mtf_trends['trend_15m'], mtf_trends['trend_1h']]
        
        # All trends should be in same direction (all positive or all negative)
        if all(t > 0 for t in trends) or all(t < 0 for t in trends):
            return True
        
        return False
    
    @staticmethod
    def calculate_features(df: pd.DataFrame, current_index: int, lookback: int = 20) -> Dict[str, Any]:
        """
        Calculate all trend features for a specific candle.
        
        Args:
            df: DataFrame with OHLC data
            current_index: Index of current candle
            lookback: Lookback period
            
        Returns:
            Dictionary of trend features
        """
        if current_index < lookback:
            return TrendAnalyzer._get_default_features()
        
        # Use data up to current index
        historical_df = df.iloc[:current_index + 1]
        
        # Single timeframe analysis
        single_tf = TrendAnalyzer.analyze_single_timeframe(historical_df, lookback)
        
        # Multi-timeframe analysis
        mtf_trends = TrendAnalyzer.multi_timeframe_analysis(historical_df)
        
        # Combine features
        features = {
            **single_tf,
            **mtf_trends,
            'trend_alignment': 1 if TrendAnalyzer.check_trend_alignment(mtf_trends) else 0
        }
        
        return features
    
    @staticmethod
    def _get_default_features() -> Dict[str, Any]:
        """Get default features when insufficient data."""
        return {
            'higher_highs_count': 0,
            'higher_lows_count': 0,
            'lower_highs_count': 0,
            'lower_lows_count': 0,
            'trend_strength': 0.0,
            'trend_direction': 0,
            'trend_5m': 0,
            'trend_15m': 0,
            'trend_1h': 0,
            'trend_alignment': 0
        }
