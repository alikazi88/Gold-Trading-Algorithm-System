"""
Support and Resistance detection for gold scalping.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from scipy.signal import argrelextrema


class SupportResistanceDetector:
    """Detect dynamic support and resistance levels."""
    
    def __init__(self, swing_window: int = 10, tolerance_pips: float = 5.0):
        """
        Initialize S/R detector.
        
        Args:
            swing_window: Window for swing point detection
            tolerance_pips: Tolerance for level clustering (in pips)
        """
        self.swing_window = swing_window
        self.tolerance = tolerance_pips * 0.1  # Convert pips to price for gold
    
    def find_swing_points(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find swing highs and lows.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Tuple of (swing_highs_indices, swing_lows_indices)
        """
        # Find local maxima (swing highs)
        swing_highs = argrelextrema(
            df['high'].values,
            np.greater,
            order=self.swing_window
        )[0]
        
        # Find local minima (swing lows)
        swing_lows = argrelextrema(
            df['low'].values,
            np.less,
            order=self.swing_window
        )[0]
        
        return swing_highs, swing_lows
    
    def cluster_levels(self, levels: List[float]) -> List[Dict[str, Any]]:
        """
        Cluster nearby levels into zones.
        
        Args:
            levels: List of price levels
            
        Returns:
            List of clustered level dictionaries
        """
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level - current_cluster[-1] <= self.tolerance:
                current_cluster.append(level)
            else:
                # Finalize current cluster
                clusters.append({
                    'level': np.mean(current_cluster),
                    'touches': len(current_cluster),
                    'strength': len(current_cluster) / len(levels)
                })
                current_cluster = [level]
        
        # Add last cluster
        if current_cluster:
            clusters.append({
                'level': np.mean(current_cluster),
                'touches': len(current_cluster),
                'strength': len(current_cluster) / len(levels)
            })
        
        return clusters
    
    def calculate_sr_levels(self, df: pd.DataFrame, lookback: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate support and resistance levels.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of candles to look back
            
        Returns:
            Dictionary with 'support' and 'resistance' level lists
        """
        if len(df) < lookback:
            lookback = len(df)
        
        recent_df = df.tail(lookback).reset_index(drop=True)
        
        swing_highs, swing_lows = self.find_swing_points(recent_df)
        
        # Extract resistance levels from swing highs
        resistance_levels = recent_df.loc[swing_highs, 'high'].tolist()
        resistance_clusters = self.cluster_levels(resistance_levels)
        
        # Extract support levels from swing lows
        support_levels = recent_df.loc[swing_lows, 'low'].tolist()
        support_clusters = self.cluster_levels(support_levels)
        
        return {
            'support': support_clusters,
            'resistance': resistance_clusters
        }
    
    def find_nearest_levels(self, current_price: float, 
                           sr_levels: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Find nearest support and resistance levels.
        
        Args:
            current_price: Current market price
            sr_levels: Dictionary of S/R levels
            
        Returns:
            Dictionary with nearest levels info
        """
        nearest_support = None
        nearest_resistance = None
        
        support_distance = float('inf')
        resistance_distance = float('inf')
        
        # Find nearest support (below current price)
        for level_info in sr_levels['support']:
            level = level_info['level']
            if level < current_price:
                distance = current_price - level
                if distance < support_distance:
                    support_distance = distance
                    nearest_support = level_info
        
        # Find nearest resistance (above current price)
        for level_info in sr_levels['resistance']:
            level = level_info['level']
            if level > current_price:
                distance = level - current_price
                if distance < resistance_distance:
                    resistance_distance = distance
                    nearest_resistance = level_info
        
        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance': support_distance if nearest_support else None,
            'resistance_distance': resistance_distance if nearest_resistance else None
        }
    
    def is_at_level(self, current_price: float, level: float, 
                   tolerance_multiplier: float = 2.0) -> bool:
        """
        Check if price is at a specific level.
        
        Args:
            current_price: Current price
            level: S/R level
            tolerance_multiplier: Multiplier for tolerance
            
        Returns:
            True if price is at the level
        """
        return abs(current_price - level) <= (self.tolerance * tolerance_multiplier)
    
    def calculate_features(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """
        Calculate S/R features for a specific candle.
        
        Args:
            df: DataFrame with OHLC data
            current_index: Index of current candle
            
        Returns:
            Dictionary of S/R features
        """
        if current_index < self.swing_window:
            return self._get_default_features()
        
        # Use data up to current index
        historical_df = df.iloc[:current_index + 1]
        
        # Calculate S/R levels
        sr_levels = self.calculate_sr_levels(historical_df, lookback=100)
        
        current_price = df.loc[current_index, 'close']
        nearest_levels = self.find_nearest_levels(current_price, sr_levels)
        
        # Extract features
        features = {
            'distance_to_support': nearest_levels['support_distance'] / current_price if nearest_levels['support_distance'] else 1.0,
            'distance_to_resistance': nearest_levels['resistance_distance'] / current_price if nearest_levels['resistance_distance'] else 1.0,
            'support_touches': nearest_levels['nearest_support']['touches'] if nearest_levels['nearest_support'] else 0,
            'resistance_touches': nearest_levels['nearest_resistance']['touches'] if nearest_levels['nearest_resistance'] else 0,
            'at_support': 1 if nearest_levels['nearest_support'] and self.is_at_level(
                current_price, nearest_levels['nearest_support']['level']
            ) else 0,
            'at_resistance': 1 if nearest_levels['nearest_resistance'] and self.is_at_level(
                current_price, nearest_levels['nearest_resistance']['level']
            ) else 0,
            'support_strength': nearest_levels['nearest_support']['strength'] if nearest_levels['nearest_support'] else 0.0,
            'resistance_strength': nearest_levels['nearest_resistance']['strength'] if nearest_levels['nearest_resistance'] else 0.0,
        }
        
        return features
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Get default features when insufficient data."""
        return {
            'distance_to_support': 1.0,
            'distance_to_resistance': 1.0,
            'support_touches': 0,
            'resistance_touches': 0,
            'at_support': 0,
            'at_resistance': 0,
            'support_strength': 0.0,
            'resistance_strength': 0.0,
        }
    
    def detect_breakout(self, df: pd.DataFrame, current_index: int) -> Dict[str, bool]:
        """
        Detect breakout of S/R levels.
        
        Args:
            df: DataFrame with OHLC data
            current_index: Index of current candle
            
        Returns:
            Dictionary with breakout flags
        """
        if current_index < 2:
            return {'resistance_breakout': False, 'support_breakout': False}
        
        current_close = df.loc[current_index, 'close']
        prev_close = df.loc[current_index - 1, 'close']
        
        historical_df = df.iloc[:current_index]
        sr_levels = self.calculate_sr_levels(historical_df, lookback=50)
        
        resistance_breakout = False
        support_breakout = False
        
        # Check resistance breakout
        for level_info in sr_levels['resistance']:
            level = level_info['level']
            if prev_close < level and current_close > level:
                resistance_breakout = True
                break
        
        # Check support breakout
        for level_info in sr_levels['support']:
            level = level_info['level']
            if prev_close > level and current_close < level:
                support_breakout = True
                break
        
        return {
            'resistance_breakout': resistance_breakout,
            'support_breakout': support_breakout
        }
