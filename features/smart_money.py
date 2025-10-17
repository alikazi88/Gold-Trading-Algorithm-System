"""
Smart Money Concepts detection for gold scalping.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from utils.helpers import get_session, encode_session


class SmartMoneyDetector:
    """Detect smart money patterns and institutional activity."""
    
    def __init__(self, volume_spike_threshold: float = 2.0):
        """
        Initialize smart money detector.
        
        Args:
            volume_spike_threshold: Multiplier for volume spike detection
        """
        self.volume_spike_threshold = volume_spike_threshold
    
    def detect_liquidity_sweep(self, df: pd.DataFrame, current_index: int, 
                               lookback: int = 10) -> Dict[str, bool]:
        """
        Detect liquidity sweeps (stop hunts).
        
        Args:
            df: DataFrame with OHLC data
            current_index: Current candle index
            lookback: Lookback period for recent highs/lows
            
        Returns:
            Dictionary with sweep flags
        """
        if current_index < lookback + 1:
            return {'liquidity_sweep_bull': False, 'liquidity_sweep_bear': False}
        
        # Get recent data
        start_idx = max(0, current_index - lookback)
        recent_df = df.iloc[start_idx:current_index]
        current_candle = df.iloc[current_index]
        prev_candle = df.iloc[current_index - 1]
        
        # Find recent high and low
        recent_high = recent_df['high'].max()
        recent_low = recent_df['low'].min()
        
        # Bullish liquidity sweep: Price breaks below recent low then closes above it
        liquidity_sweep_bull = (
            current_candle['low'] < recent_low and
            current_candle['close'] > recent_low and
            current_candle['close'] > current_candle['open']
        )
        
        # Bearish liquidity sweep: Price breaks above recent high then closes below it
        liquidity_sweep_bear = (
            current_candle['high'] > recent_high and
            current_candle['close'] < recent_high and
            current_candle['close'] < current_candle['open']
        )
        
        return {
            'liquidity_sweep_bull': liquidity_sweep_bull,
            'liquidity_sweep_bear': liquidity_sweep_bear
        }
    
    def detect_institutional_candle(self, df: pd.DataFrame, current_index: int,
                                   body_threshold: float = 0.7) -> bool:
        """
        Detect institutional candle patterns (large body with immediate reversal).
        
        Args:
            df: DataFrame with OHLC data
            current_index: Current candle index
            body_threshold: Minimum body to range ratio
            
        Returns:
            True if institutional candle detected
        """
        if current_index < 1:
            return False
        
        current = df.iloc[current_index]
        prev = df.iloc[current_index - 1]
        
        # Calculate body and range
        current_body = abs(current['close'] - current['open'])
        current_range = current['high'] - current['low']
        prev_body = abs(prev['close'] - prev['open'])
        prev_range = prev['high'] - prev['low']
        
        # Large body candle
        if current_range > 0:
            body_ratio = current_body / current_range
        else:
            return False
        
        # Check for large body
        is_large_body = body_ratio > body_threshold
        
        # Check for size relative to previous candle
        is_larger = current_body > prev_body * 1.5
        
        # Check for immediate reversal (next candle if available)
        has_reversal = False
        if current_index < len(df) - 1:
            next_candle = df.iloc[current_index + 1]
            # Reversal: bullish candle followed by bearish or vice versa
            current_bullish = current['close'] > current['open']
            next_bullish = next_candle['close'] > next_candle['open']
            has_reversal = current_bullish != next_bullish
        
        return is_large_body and is_larger
    
    def detect_volume_spike(self, df: pd.DataFrame, current_index: int,
                           lookback: int = 20) -> bool:
        """
        Detect volume spikes.
        
        Args:
            df: DataFrame with volume data
            current_index: Current candle index
            lookback: Lookback period for average volume
            
        Returns:
            True if volume spike detected
        """
        if current_index < lookback or 'volume' not in df.columns:
            return False
        
        start_idx = max(0, current_index - lookback)
        recent_volume = df.iloc[start_idx:current_index]['volume']
        avg_volume = recent_volume.mean()
        
        if avg_volume == 0:
            return False
        
        current_volume = df.iloc[current_index]['volume']
        
        return current_volume > (avg_volume * self.volume_spike_threshold)
    
    def get_session_features(self, timestamp: int) -> Dict[str, int]:
        """
        Get trading session features.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Dictionary with session features
        """
        from datetime import datetime
        dt = datetime.utcfromtimestamp(timestamp)
        session = get_session(dt)
        
        asian, london, new_york, overlap = encode_session(session)
        
        return {
            'session_asian': asian,
            'session_london': london,
            'session_new_york': new_york,
            'session_overlap': overlap
        }
    
    def detect_accumulation_distribution(self, df: pd.DataFrame, current_index: int,
                                        lookback: int = 10) -> int:
        """
        Detect accumulation (bullish) or distribution (bearish) patterns.
        
        Args:
            df: DataFrame with OHLC data
            current_index: Current candle index
            lookback: Lookback period
            
        Returns:
            1 for accumulation, -1 for distribution, 0 for neutral
        """
        if current_index < lookback:
            return 0
        
        start_idx = max(0, current_index - lookback)
        recent_df = df.iloc[start_idx:current_index + 1]
        
        # Calculate buying/selling pressure
        buying_pressure = 0
        selling_pressure = 0
        
        for idx in range(len(recent_df)):
            candle = recent_df.iloc[idx]
            candle_range = candle['high'] - candle['low']
            
            if candle_range == 0:
                continue
            
            # Where did price close in the range?
            close_position = (candle['close'] - candle['low']) / candle_range
            
            if close_position > 0.6:  # Closed in upper 40%
                buying_pressure += 1
            elif close_position < 0.4:  # Closed in lower 40%
                selling_pressure += 1
        
        # Determine pattern
        if buying_pressure > selling_pressure * 1.5:
            return 1  # Accumulation
        elif selling_pressure > buying_pressure * 1.5:
            return -1  # Distribution
        else:
            return 0  # Neutral
    
    def calculate_features(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """
        Calculate all smart money features.
        
        Args:
            df: DataFrame with OHLC data
            current_index: Current candle index
            
        Returns:
            Dictionary of smart money features
        """
        if current_index < 10:
            return self._get_default_features(df, current_index)
        
        # Liquidity sweeps
        sweeps = self.detect_liquidity_sweep(df, current_index)
        
        # Institutional candle
        institutional = self.detect_institutional_candle(df, current_index)
        
        # Volume spike
        volume_spike = self.detect_volume_spike(df, current_index)
        
        # Session features
        timestamp = df.iloc[current_index]['timestamp']
        session_features = self.get_session_features(timestamp)
        
        # Accumulation/Distribution
        acc_dist = self.detect_accumulation_distribution(df, current_index)
        
        features = {
            'liquidity_sweep_bull': int(sweeps['liquidity_sweep_bull']),
            'liquidity_sweep_bear': int(sweeps['liquidity_sweep_bear']),
            'institutional_candle': int(institutional),
            'volume_spike': int(volume_spike),
            **session_features,
            'accumulation_distribution': acc_dist
        }
        
        return features
    
    def _get_default_features(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """Get default features when insufficient data."""
        timestamp = df.iloc[current_index]['timestamp'] if current_index < len(df) else 0
        session_features = self.get_session_features(timestamp) if timestamp > 0 else {
            'session_asian': 0,
            'session_london': 0,
            'session_new_york': 0,
            'session_overlap': 0
        }
        
        return {
            'liquidity_sweep_bull': 0,
            'liquidity_sweep_bear': 0,
            'institutional_candle': 0,
            'volume_spike': 0,
            **session_features,
            'accumulation_distribution': 0
        }
