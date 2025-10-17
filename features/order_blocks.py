"""
Order Block detection and tracking for gold scalping.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class OrderBlockDetector:
    """Detect and track order blocks."""
    
    def __init__(self, strength_threshold: float = 0.6, min_move_pips: float = 10.0):
        """
        Initialize order block detector.
        
        Args:
            strength_threshold: Minimum strength score for valid order block
            min_move_pips: Minimum price move after OB to be considered valid
        """
        self.strength_threshold = strength_threshold
        self.min_move = min_move_pips * 0.1  # Convert pips to price for gold
        self.active_order_blocks = []
    
    def detect_order_block(self, df: pd.DataFrame, current_index: int,
                          lookback: int = 10) -> Optional[Dict[str, Any]]:
        """
        Detect order block formation.
        
        An order block is the last opposing candle before a strong directional move.
        
        Bullish OB: Last bearish candle before strong upward move
        Bearish OB: Last bullish candle before strong downward move
        
        Args:
            df: DataFrame with OHLC data
            current_index: Current candle index
            lookback: Period to check for strong move
            
        Returns:
            Order block dictionary or None
        """
        if current_index < lookback + 2:
            return None
        
        # Check for strong move in recent candles
        start_idx = current_index - lookback
        recent_df = df.iloc[start_idx:current_index + 1]
        
        # Calculate net move
        start_price = recent_df.iloc[0]['close']
        end_price = recent_df.iloc[-1]['close']
        net_move = abs(end_price - start_price)
        
        if net_move < self.min_move:
            return None
        
        # Determine move direction
        is_bullish_move = end_price > start_price
        
        # Find the last opposing candle before the move
        order_block_candle = None
        ob_index = None
        
        if is_bullish_move:
            # Look for last bearish candle
            for i in range(len(recent_df) - 1, -1, -1):
                candle = recent_df.iloc[i]
                if candle['close'] < candle['open']:  # Bearish candle
                    order_block_candle = candle
                    ob_index = start_idx + i
                    break
        else:
            # Look for last bullish candle
            for i in range(len(recent_df) - 1, -1, -1):
                candle = recent_df.iloc[i]
                if candle['close'] > candle['open']:  # Bullish candle
                    order_block_candle = candle
                    ob_index = start_idx + i
                    break
        
        if order_block_candle is None:
            return None
        
        # Calculate order block strength based on subsequent move
        strength = min(net_move / (order_block_candle['high'] - order_block_candle['low']), 2.0) / 2.0
        
        if strength < self.strength_threshold:
            return None
        
        ob_type = 'bullish' if is_bullish_move else 'bearish'
        
        return {
            'type': ob_type,
            'high': order_block_candle['high'],
            'low': order_block_candle['low'],
            'open': order_block_candle['open'],
            'close': order_block_candle['close'],
            'index': ob_index,
            'timestamp': order_block_candle['timestamp'],
            'strength': strength,
            'mitigated': False,
            'subsequent_move': net_move
        }
    
    def check_mitigation(self, ob: Dict[str, Any], current_candle: pd.Series) -> bool:
        """
        Check if order block has been mitigated (price returned to test it).
        
        Args:
            ob: Order block dictionary
            current_candle: Current candle data
            
        Returns:
            True if order block is mitigated
        """
        if ob['type'] == 'bullish':
            # Bullish OB is mitigated when price returns to its zone
            return current_candle['low'] <= ob['high'] and current_candle['low'] >= ob['low']
        else:
            # Bearish OB is mitigated when price returns to its zone
            return current_candle['high'] >= ob['low'] and current_candle['high'] <= ob['high']
    
    def update_active_order_blocks(self, df: pd.DataFrame, current_index: int) -> None:
        """
        Update list of active order blocks.
        
        Args:
            df: DataFrame with OHLC data
            current_index: Current candle index
        """
        # Detect new order block
        new_ob = self.detect_order_block(df, current_index)
        if new_ob:
            # Check if similar OB already exists
            is_duplicate = False
            for existing_ob in self.active_order_blocks:
                if (existing_ob['type'] == new_ob['type'] and
                    abs(existing_ob['high'] - new_ob['high']) < self.min_move and
                    abs(existing_ob['low'] - new_ob['low']) < self.min_move):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                self.active_order_blocks.append(new_ob)
        
        # Update existing order blocks
        current_candle = df.iloc[current_index]
        for ob in self.active_order_blocks:
            if not ob['mitigated']:
                if self.check_mitigation(ob, current_candle):
                    ob['mitigated'] = True
                    ob['mitigated_index'] = current_index
        
        # Remove old mitigated order blocks (keep last 50)
        if len(self.active_order_blocks) > 50:
            self.active_order_blocks = self.active_order_blocks[-50:]
    
    def get_nearest_order_block(self, current_price: float, 
                               ob_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get nearest unmitigated order block.
        
        Args:
            current_price: Current market price
            ob_type: Filter by type ('bullish' or 'bearish'), None for any
            
        Returns:
            Nearest order block or None
        """
        unmitigated_obs = [ob for ob in self.active_order_blocks if not ob['mitigated']]
        
        if ob_type:
            unmitigated_obs = [ob for ob in unmitigated_obs if ob['type'] == ob_type]
        
        if not unmitigated_obs:
            return None
        
        # Find nearest by distance to OB zone
        def distance_to_ob(ob):
            if ob['low'] <= current_price <= ob['high']:
                return 0.0  # Inside OB
            elif current_price < ob['low']:
                return ob['low'] - current_price
            else:
                return current_price - ob['high']
        
        nearest = min(unmitigated_obs, key=distance_to_ob)
        return nearest
    
    def calculate_distance_to_ob(self, current_price: float, ob: Dict[str, Any]) -> float:
        """
        Calculate distance from current price to order block.
        
        Args:
            current_price: Current market price
            ob: Order block dictionary
            
        Returns:
            Distance to order block
        """
        if ob['low'] <= current_price <= ob['high']:
            return 0.0  # Inside OB
        elif current_price < ob['low']:
            return ob['low'] - current_price
        else:
            return current_price - ob['high']
    
    def is_price_in_ob(self, current_price: float, ob: Dict[str, Any]) -> bool:
        """
        Check if price is currently inside order block zone.
        
        Args:
            current_price: Current market price
            ob: Order block dictionary
            
        Returns:
            True if price is in OB
        """
        return ob['low'] <= current_price <= ob['high']
    
    def calculate_features(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """
        Calculate order block features for current candle.
        
        Args:
            df: DataFrame with OHLC data
            current_index: Current candle index
            
        Returns:
            Dictionary of order block features
        """
        if current_index < 12:
            return self._get_default_features()
        
        # Update active order blocks
        self.update_active_order_blocks(df, current_index)
        
        current_price = df.iloc[current_index]['close']
        
        # Get nearest order blocks
        nearest_bull_ob = self.get_nearest_order_block(current_price, 'bullish')
        nearest_bear_ob = self.get_nearest_order_block(current_price, 'bearish')
        
        # Calculate features
        features = {
            'ob_bull_present': 1 if nearest_bull_ob else 0,
            'ob_bear_present': 1 if nearest_bear_ob else 0,
        }
        
        if nearest_bull_ob:
            features['ob_bull_distance'] = self.calculate_distance_to_ob(current_price, nearest_bull_ob)
            features['ob_bull_strength'] = nearest_bull_ob['strength']
            features['ob_bull_mitigated'] = 1 if nearest_bull_ob['mitigated'] else 0
            features['price_in_bull_ob'] = 1 if self.is_price_in_ob(current_price, nearest_bull_ob) else 0
        else:
            features['ob_bull_distance'] = 999.0
            features['ob_bull_strength'] = 0.0
            features['ob_bull_mitigated'] = 0
            features['price_in_bull_ob'] = 0
        
        if nearest_bear_ob:
            features['ob_bear_distance'] = self.calculate_distance_to_ob(current_price, nearest_bear_ob)
            features['ob_bear_strength'] = nearest_bear_ob['strength']
            features['ob_bear_mitigated'] = 1 if nearest_bear_ob['mitigated'] else 0
            features['price_in_bear_ob'] = 1 if self.is_price_in_ob(current_price, nearest_bear_ob) else 0
        else:
            features['ob_bear_distance'] = 999.0
            features['ob_bear_strength'] = 0.0
            features['ob_bear_mitigated'] = 0
            features['price_in_bear_ob'] = 0
        
        return features
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Get default features when insufficient data."""
        return {
            'ob_bull_present': 0,
            'ob_bear_present': 0,
            'ob_bull_distance': 999.0,
            'ob_bear_distance': 999.0,
            'ob_bull_strength': 0.0,
            'ob_bear_strength': 0.0,
            'ob_bull_mitigated': 0,
            'ob_bear_mitigated': 0,
            'price_in_bull_ob': 0,
            'price_in_bear_ob': 0
        }
    
    def reset(self) -> None:
        """Reset active order blocks list."""
        self.active_order_blocks = []
