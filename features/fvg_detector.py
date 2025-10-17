"""
Fair Value Gap (FVG) detection for gold scalping.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class FVGDetector:
    """Detect and track Fair Value Gaps."""
    
    def __init__(self, min_size_pips: float = 5.0):
        """
        Initialize FVG detector.
        
        Args:
            min_size_pips: Minimum FVG size in pips to be considered valid
        """
        self.min_size = min_size_pips * 0.1  # Convert pips to price for gold
        self.active_fvgs = []
    
    def detect_fvg(self, df: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        """
        Detect FVG at current candle.
        
        A bullish FVG occurs when:
        - Candle[i-2].low > Candle[i].high (gap between them)
        
        A bearish FVG occurs when:
        - Candle[i-2].high < Candle[i].low (gap between them)
        
        Args:
            df: DataFrame with OHLC data
            current_index: Current candle index
            
        Returns:
            FVG dictionary or None
        """
        if current_index < 2:
            return None
        
        candle_0 = df.iloc[current_index - 2]  # Two candles ago
        candle_1 = df.iloc[current_index - 1]  # Previous candle
        candle_2 = df.iloc[current_index]      # Current candle
        
        # Bullish FVG: Gap between candle_0.low and candle_2.high
        if candle_0['low'] > candle_2['high']:
            gap_size = candle_0['low'] - candle_2['high']
            
            if gap_size >= self.min_size:
                return {
                    'type': 'bullish',
                    'upper_level': candle_0['low'],
                    'lower_level': candle_2['high'],
                    'size': gap_size,
                    'created_index': current_index,
                    'created_timestamp': df.iloc[current_index]['timestamp'],
                    'mitigated': False
                }
        
        # Bearish FVG: Gap between candle_0.high and candle_2.low
        elif candle_0['high'] < candle_2['low']:
            gap_size = candle_2['low'] - candle_0['high']
            
            if gap_size >= self.min_size:
                return {
                    'type': 'bearish',
                    'upper_level': candle_2['low'],
                    'lower_level': candle_0['high'],
                    'size': gap_size,
                    'created_index': current_index,
                    'created_timestamp': df.iloc[current_index]['timestamp'],
                    'mitigated': False
                }
        
        return None
    
    def check_fvg_mitigation(self, fvg: Dict[str, Any], current_candle: pd.Series) -> bool:
        """
        Check if FVG has been filled/mitigated.
        
        Args:
            fvg: FVG dictionary
            current_candle: Current candle data
            
        Returns:
            True if FVG is mitigated
        """
        if fvg['type'] == 'bullish':
            # Bullish FVG is mitigated when price trades back down into the gap
            return current_candle['low'] <= fvg['upper_level']
        else:
            # Bearish FVG is mitigated when price trades back up into the gap
            return current_candle['high'] >= fvg['lower_level']
    
    def update_active_fvgs(self, df: pd.DataFrame, current_index: int) -> None:
        """
        Update list of active FVGs.
        
        Args:
            df: DataFrame with OHLC data
            current_index: Current candle index
        """
        current_candle = df.iloc[current_index]
        
        # Check for new FVG
        new_fvg = self.detect_fvg(df, current_index)
        if new_fvg:
            self.active_fvgs.append(new_fvg)
        
        # Update existing FVGs
        for fvg in self.active_fvgs:
            if not fvg['mitigated']:
                if self.check_fvg_mitigation(fvg, current_candle):
                    fvg['mitigated'] = True
                    fvg['mitigated_index'] = current_index
    
    def get_nearest_unfilled_fvg(self, current_price: float, fvg_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get nearest unfilled FVG.
        
        Args:
            current_price: Current market price
            fvg_type: Filter by type ('bullish' or 'bearish'), None for any
            
        Returns:
            Nearest unfilled FVG or None
        """
        unfilled_fvgs = [fvg for fvg in self.active_fvgs if not fvg['mitigated']]
        
        if fvg_type:
            unfilled_fvgs = [fvg for fvg in unfilled_fvgs if fvg['type'] == fvg_type]
        
        if not unfilled_fvgs:
            return None
        
        # Find nearest by distance to current price
        nearest = min(unfilled_fvgs, key=lambda fvg: min(
            abs(current_price - fvg['upper_level']),
            abs(current_price - fvg['lower_level'])
        ))
        
        return nearest
    
    def calculate_distance_to_fvg(self, current_price: float, fvg: Dict[str, Any]) -> float:
        """
        Calculate distance from current price to FVG.
        
        Args:
            current_price: Current market price
            fvg: FVG dictionary
            
        Returns:
            Distance to FVG
        """
        if fvg['type'] == 'bullish':
            # Distance to upper level of bullish FVG
            if current_price < fvg['lower_level']:
                return fvg['lower_level'] - current_price
            elif current_price > fvg['upper_level']:
                return current_price - fvg['upper_level']
            else:
                return 0.0  # Inside FVG
        else:
            # Distance to lower level of bearish FVG
            if current_price > fvg['upper_level']:
                return current_price - fvg['upper_level']
            elif current_price < fvg['lower_level']:
                return fvg['lower_level'] - current_price
            else:
                return 0.0  # Inside FVG
    
    def is_price_in_fvg(self, current_price: float, fvg: Dict[str, Any]) -> bool:
        """
        Check if price is currently inside FVG.
        
        Args:
            current_price: Current market price
            fvg: FVG dictionary
            
        Returns:
            True if price is in FVG
        """
        return fvg['lower_level'] <= current_price <= fvg['upper_level']
    
    def calculate_features(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """
        Calculate FVG features for current candle.
        
        Args:
            df: DataFrame with OHLC data
            current_index: Current candle index
            
        Returns:
            Dictionary of FVG features
        """
        if current_index < 2:
            return self._get_default_features()
        
        # Update active FVGs
        self.update_active_fvgs(df, current_index)
        
        current_price = df.iloc[current_index]['close']
        current_timestamp = df.iloc[current_index]['timestamp']
        
        # Get nearest unfilled FVGs
        nearest_bull_fvg = self.get_nearest_unfilled_fvg(current_price, 'bullish')
        nearest_bear_fvg = self.get_nearest_unfilled_fvg(current_price, 'bearish')
        nearest_any_fvg = self.get_nearest_unfilled_fvg(current_price)
        
        # Calculate features
        features = {
            'fvg_present': 1 if nearest_any_fvg else 0,
            'fvg_bullish': 1 if nearest_bull_fvg else 0,
            'fvg_bearish': 1 if nearest_bear_fvg else 0,
        }
        
        if nearest_any_fvg:
            features['fvg_size'] = nearest_any_fvg['size']
            features['fvg_distance'] = self.calculate_distance_to_fvg(current_price, nearest_any_fvg)
            
            # Age in candles
            features['fvg_age'] = current_index - nearest_any_fvg['created_index']
            
            # Is price in FVG?
            features['price_in_fvg'] = 1 if self.is_price_in_fvg(current_price, nearest_any_fvg) else 0
        else:
            features['fvg_size'] = 0.0
            features['fvg_distance'] = 999.0  # Large value indicating no FVG
            features['fvg_age'] = 0
            features['price_in_fvg'] = 0
        
        return features
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Get default features when insufficient data."""
        return {
            'fvg_present': 0,
            'fvg_bullish': 0,
            'fvg_bearish': 0,
            'fvg_size': 0.0,
            'fvg_distance': 999.0,
            'fvg_age': 0,
            'price_in_fvg': 0
        }
    
    def reset(self) -> None:
        """Reset active FVGs list."""
        self.active_fvgs = []
