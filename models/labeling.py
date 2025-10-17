"""
Label historical data for supervised learning.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from utils.helpers import price_to_pips
from utils.logger import TradingLogger


class TradeLabeler:
    """Label candles based on future price action for ML training."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[TradingLogger] = None):
        """
        Initialize trade labeler.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or TradingLogger.get_logger(__name__)
        
        labeling_config = config.get('labeling', {})
        risk_config = config.get('risk', {})
        
        self.target_rr = labeling_config.get('target_rr', 2.0)
        self.max_holding_candles = labeling_config.get('max_holding_candles', 20)
        self.atr_period = risk_config.get('atr_period', 20)
        self.sl_atr_multiplier = risk_config.get('stop_loss_atr_multiplier', 1.0)
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def label_candle(self, df: pd.DataFrame, index: int, atr: pd.Series) -> int:
        """
        Label a single candle based on future price action.
        
        Returns:
            1 for BUY, -1 for SELL, 0 for NO_TRADE
        
        Args:
            df: DataFrame with OHLC data
            index: Index of candle to label
            atr: ATR series
            
        Returns:
            Label (1, -1, or 0)
        """
        if index >= len(df) - self.max_holding_candles:
            return 0  # Not enough future data
        
        current_price = df.iloc[index]['close']
        current_atr = atr.iloc[index]
        
        if pd.isna(current_atr) or current_atr == 0:
            return 0
        
        # Calculate stop loss and take profit distances
        sl_distance = current_atr * self.sl_atr_multiplier
        tp_distance = sl_distance * self.target_rr
        
        # Define levels for BUY
        buy_sl = current_price - sl_distance
        buy_tp = current_price + tp_distance
        
        # Define levels for SELL
        sell_sl = current_price + sl_distance
        sell_tp = current_price - tp_distance
        
        # Check future price action
        future_df = df.iloc[index + 1:index + 1 + self.max_holding_candles]
        
        # Check BUY scenario
        buy_hit_sl = (future_df['low'] <= buy_sl).any()
        buy_hit_tp = (future_df['high'] >= buy_tp).any()
        
        if buy_hit_tp and not buy_hit_sl:
            # TP hit before SL
            return 1
        elif buy_hit_tp and buy_hit_sl:
            # Both hit, check which came first
            sl_index = future_df[future_df['low'] <= buy_sl].index[0]
            tp_index = future_df[future_df['high'] >= buy_tp].index[0]
            if tp_index < sl_index:
                return 1
        
        # Check SELL scenario
        sell_hit_sl = (future_df['high'] >= sell_sl).any()
        sell_hit_tp = (future_df['low'] <= sell_tp).any()
        
        if sell_hit_tp and not sell_hit_sl:
            # TP hit before SL
            return -1
        elif sell_hit_tp and sell_hit_sl:
            # Both hit, check which came first
            sl_index = future_df[future_df['high'] >= sell_sl].index[0]
            tp_index = future_df[future_df['low'] <= sell_tp].index[0]
            if tp_index < sl_index:
                return -1
        
        # Neither scenario successful
        return 0
    
    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label entire dataset.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with 'label' column added
        """
        self.logger.info(f"Labeling {len(df)} candles")
        
        # Calculate ATR
        atr = self.calculate_atr(df)
        
        # Initialize label column
        df['label'] = 0
        
        # Label each candle
        for idx in range(len(df) - self.max_holding_candles):
            try:
                label = self.label_candle(df, idx, atr)
                df.at[idx, 'label'] = label
            except Exception as e:
                self.logger.warning(f"Error labeling candle at index {idx}: {e}")
                continue
        
        # Log label distribution
        label_counts = df['label'].value_counts()
        self.logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        buy_pct = (label_counts.get(1, 0) / len(df)) * 100
        sell_pct = (label_counts.get(-1, 0) / len(df)) * 100
        no_trade_pct = (label_counts.get(0, 0) / len(df)) * 100
        
        self.logger.info(f"BUY: {buy_pct:.2f}%, SELL: {sell_pct:.2f}%, NO_TRADE: {no_trade_pct:.2f}%")
        
        return df
    
    def get_label_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about labels.
        
        Args:
            df: DataFrame with labels
            
        Returns:
            Dictionary with statistics
        """
        if 'label' not in df.columns:
            return {}
        
        label_counts = df['label'].value_counts()
        total = len(df)
        
        return {
            'total_samples': total,
            'buy_count': int(label_counts.get(1, 0)),
            'sell_count': int(label_counts.get(-1, 0)),
            'no_trade_count': int(label_counts.get(0, 0)),
            'buy_percentage': (label_counts.get(1, 0) / total) * 100,
            'sell_percentage': (label_counts.get(-1, 0) / total) * 100,
            'no_trade_percentage': (label_counts.get(0, 0) / total) * 100,
            'tradeable_percentage': ((label_counts.get(1, 0) + label_counts.get(-1, 0)) / total) * 100
        }
