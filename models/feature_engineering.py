"""
Combine all feature engineering modules for ML model.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from features.price_action import PriceActionFeatures
from features.support_resistance import SupportResistanceDetector
from features.trend_analysis import TrendAnalyzer
from features.smart_money import SmartMoneyDetector
from features.fvg_detector import FVGDetector
from features.order_blocks import OrderBlockDetector
from utils.logger import TradingLogger


class FeatureEngineer:
    """Orchestrate all feature engineering for the ML model."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[TradingLogger] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or TradingLogger.get_logger(__name__)
        
        feature_config = config.get('features', {})
        
        # Initialize feature detectors
        self.sr_detector = SupportResistanceDetector(
            swing_window=feature_config.get('swing_detection_window', 10),
            tolerance_pips=5.0
        )
        
        self.smart_money = SmartMoneyDetector(
            volume_spike_threshold=feature_config.get('volume_spike_threshold', 2.0)
        )
        
        self.fvg_detector = FVGDetector(
            min_size_pips=feature_config.get('fvg_min_size_pips', 5.0)
        )
        
        self.order_block_detector = OrderBlockDetector(
            strength_threshold=feature_config.get('order_block_strength_threshold', 0.6),
            min_move_pips=10.0
        )
        
        self.lookback = feature_config.get('lookback_candles', 20)
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features for entire DataFrame.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all features
        """
        self.logger.info(f"Calculating features for {len(df)} candles")
        
        # Reset detectors
        self.fvg_detector.reset()
        self.order_block_detector.reset()
        
        # Calculate price action features (vectorized)
        df = PriceActionFeatures.calculate_all_features(df, self.lookback)
        
        # Initialize feature columns
        feature_columns = self._get_feature_column_names()
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Calculate features for each candle (sequential for stateful detectors)
        for idx in range(len(df)):
            try:
                # Support/Resistance features
                sr_features = self.sr_detector.calculate_features(df, idx)
                
                # Trend features
                trend_features = TrendAnalyzer.calculate_features(df, idx, self.lookback)
                
                # Smart Money features
                sm_features = self.smart_money.calculate_features(df, idx)
                
                # FVG features
                fvg_features = self.fvg_detector.calculate_features(df, idx)
                
                # Order Block features
                ob_features = self.order_block_detector.calculate_features(df, idx)
                
                # Combine all features
                all_features = {
                    **sr_features,
                    **trend_features,
                    **sm_features,
                    **fvg_features,
                    **ob_features
                }
                
                # Update DataFrame
                for key, value in all_features.items():
                    if key in df.columns:
                        df.at[idx, key] = value
                
            except Exception as e:
                self.logger.warning(f"Error calculating features for index {idx}: {e}")
                continue
        
        self.logger.info("Feature calculation completed")
        return df
    
    def calculate_features_for_candle(self, df: pd.DataFrame, index: int) -> Dict[str, Any]:
        """
        Calculate features for a single candle (for live trading).
        
        Args:
            df: DataFrame with OHLC data up to current candle
            index: Index of the candle
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        try:
            # Price action features (from DataFrame if already calculated)
            if 'body_size' in df.columns:
                pa_features = PriceActionFeatures.extract_features_for_ml(df, index)
                features.update(pa_features)
            
            # Support/Resistance features
            sr_features = self.sr_detector.calculate_features(df, index)
            features.update(sr_features)
            
            # Trend features
            trend_features = TrendAnalyzer.calculate_features(df, index, self.lookback)
            features.update(trend_features)
            
            # Smart Money features
            sm_features = self.smart_money.calculate_features(df, index)
            features.update(sm_features)
            
            # FVG features
            fvg_features = self.fvg_detector.calculate_features(df, index)
            features.update(fvg_features)
            
            # Order Block features
            ob_features = self.order_block_detector.calculate_features(df, index)
            features.update(ob_features)
            
        except Exception as e:
            self.logger.error(f"Error calculating features for candle: {e}")
            return {}
        
        return features
    
    def get_feature_names(self) -> list:
        """
        Get list of all feature names for ML model.
        
        Returns:
            List of feature names
        """
        return self._get_feature_column_names()
    
    def _get_feature_column_names(self) -> list:
        """Get all feature column names."""
        return [
            # Price Action
            'body_size', 'wick_size', 'candle_range', 'close_open_ratio',
            'is_doji', 'is_engulfing', 'is_pin_bar',
            'price_momentum_3', 'price_momentum_5', 'price_momentum_10',
            'distance_to_high_20', 'distance_to_low_20',
            
            # Support/Resistance
            'distance_to_support', 'distance_to_resistance',
            'support_touches', 'resistance_touches',
            'at_support', 'at_resistance',
            'support_strength', 'resistance_strength',
            
            # Trend
            'higher_highs_count', 'higher_lows_count',
            'lower_highs_count', 'lower_lows_count',
            'trend_strength', 'trend_direction',
            'trend_5m', 'trend_15m', 'trend_1h', 'trend_alignment',
            
            # Smart Money
            'liquidity_sweep_bull', 'liquidity_sweep_bear',
            'institutional_candle', 'volume_spike',
            'session_asian', 'session_london', 'session_new_york', 'session_overlap',
            'accumulation_distribution',
            
            # Fair Value Gaps
            'fvg_present', 'fvg_bullish', 'fvg_bearish',
            'fvg_size', 'fvg_distance', 'fvg_age', 'price_in_fvg',
            
            # Order Blocks
            'ob_bull_present', 'ob_bear_present',
            'ob_bull_distance', 'ob_bear_distance',
            'ob_bull_strength', 'ob_bear_strength',
            'ob_bull_mitigated', 'ob_bear_mitigated',
            'price_in_bull_ob', 'price_in_bear_ob'
        ]
    
    def prepare_features_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model (handle missing values, etc.).
        
        Args:
            df: DataFrame with features
            
        Returns:
            Cleaned DataFrame ready for model
        """
        feature_names = self.get_feature_names()
        
        # Select only feature columns that exist
        available_features = [f for f in feature_names if f in df.columns]
        df_features = df[available_features].copy()
        
        # Handle missing values
        df_features = df_features.fillna(0)
        
        # Handle infinite values
        df_features = df_features.replace([np.inf, -np.inf], 0)
        
        return df_features
