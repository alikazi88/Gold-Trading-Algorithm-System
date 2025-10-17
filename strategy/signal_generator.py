"""
Generate trading signals using trained ML model.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from utils.logger import TradingLogger
from utils.helpers import price_to_pips


class SignalGenerator:
    """Generate trading signals from ML model predictions."""
    
    def __init__(self, model, config: Dict[str, Any], logger: Optional[TradingLogger] = None):
        """
        Initialize signal generator.
        
        Args:
            model: Trained ML model
            config: Configuration dictionary
            logger: Logger instance
        """
        self.model = model
        self.config = config
        self.logger = logger or TradingLogger.get_logger(__name__)
        
        trading_config = config.get('trading', {})
        self.min_confidence = trading_config.get('min_confidence_threshold', 0.70)
        self.symbol = trading_config.get('symbol', 'XAUUSD')
    
    def generate_signal(self, features: Dict[str, Any], current_price: float,
                       atr: float) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal from features.
        
        Args:
            features: Dictionary of calculated features
            current_price: Current market price
            atr: Current ATR value
            
        Returns:
            Signal dictionary or None if no signal
        """
        try:
            # Get model prediction
            prediction, confidence = self.model.predict_single(features)
            
            # Check confidence threshold
            if confidence < self.min_confidence:
                self.logger.debug(f"Signal confidence {confidence:.2%} below threshold {self.min_confidence:.2%}")
                return None
            
            # No trade signal
            if prediction == 0:
                return None
            
            # Generate signal
            signal_type = "BUY" if prediction == 1 else "SELL"
            
            signal = {
                'timestamp': int(datetime.utcnow().timestamp()),
                'signal_type': signal_type,
                'entry_price': current_price,
                'confidence': confidence,
                'atr': atr,
                'features': features
            }
            
            self.logger.info(f"Generated {signal_type} signal at {current_price} with {confidence:.2%} confidence")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    def validate_signal(self, signal: Dict[str, Any], market_context: Dict[str, Any]) -> bool:
        """
        Validate signal against market conditions and filters.
        
        Args:
            signal: Generated signal
            market_context: Current market context
            
        Returns:
            True if signal is valid
        """
        # Check if market is open (basic check)
        # In production, you'd check actual market hours
        
        # Validate signal has required fields
        required_fields = ['signal_type', 'entry_price', 'confidence']
        if not all(field in signal for field in required_fields):
            self.logger.warning("Signal missing required fields")
            return False
        
        # Additional validation rules can be added here
        # For example:
        # - Check spread
        # - Check volatility
        # - Check news events
        # - Check correlation with other pairs
        
        return True
    
    def enhance_signal_with_context(self, signal: Dict[str, Any], 
                                    df: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhance signal with additional market context.
        
        Args:
            signal: Base signal
            df: DataFrame with recent market data
            
        Returns:
            Enhanced signal
        """
        if len(df) == 0:
            return signal
        
        latest = df.iloc[-1]
        
        # Add market context
        signal['market_context'] = {
            'current_high': float(latest['high']),
            'current_low': float(latest['low']),
            'current_close': float(latest['close']),
            'timestamp': int(latest['timestamp'])
        }
        
        # Add trend context if available
        if 'trend_direction' in latest:
            signal['trend_direction'] = int(latest['trend_direction'])
        
        # Add S/R context
        if 'at_support' in latest and 'at_resistance' in latest:
            signal['at_support'] = int(latest['at_support'])
            signal['at_resistance'] = int(latest['at_resistance'])
        
        return signal
    
    def calculate_signal_quality_score(self, signal: Dict[str, Any]) -> float:
        """
        Calculate quality score for signal (0-1).
        
        Args:
            signal: Trading signal
            
        Returns:
            Quality score
        """
        score = 0.0
        weights = {
            'confidence': 0.4,
            'trend_alignment': 0.2,
            'sr_confluence': 0.2,
            'smart_money': 0.2
        }
        
        # Confidence component
        score += signal.get('confidence', 0) * weights['confidence']
        
        # Trend alignment
        features = signal.get('features', {})
        if features.get('trend_alignment', 0) == 1:
            score += weights['trend_alignment']
        
        # S/R confluence
        signal_type = signal.get('signal_type')
        if signal_type == 'BUY' and signal.get('at_support', 0) == 1:
            score += weights['sr_confluence']
        elif signal_type == 'SELL' and signal.get('at_resistance', 0) == 1:
            score += weights['sr_confluence']
        
        # Smart money indicators
        if signal_type == 'BUY' and features.get('liquidity_sweep_bull', 0) == 1:
            score += weights['smart_money']
        elif signal_type == 'SELL' and features.get('liquidity_sweep_bear', 0) == 1:
            score += weights['smart_money']
        
        return min(score, 1.0)
    
    def filter_signals_by_time(self, signal: Dict[str, Any]) -> bool:
        """
        Filter signals based on time of day.
        
        Args:
            signal: Trading signal
            
        Returns:
            True if signal passes time filter
        """
        timestamp = signal.get('timestamp', 0)
        dt = datetime.utcfromtimestamp(timestamp)
        hour = dt.hour
        
        # Avoid low liquidity periods (example: 22:00-00:00 UTC)
        if 22 <= hour or hour < 0:
            self.logger.debug("Signal filtered: low liquidity period")
            return False
        
        return True
    
    def get_signal_summary(self, signal: Dict[str, Any]) -> str:
        """
        Get human-readable signal summary.
        
        Args:
            signal: Trading signal
            
        Returns:
            Summary string
        """
        signal_type = signal.get('signal_type', 'UNKNOWN')
        entry = signal.get('entry_price', 0)
        confidence = signal.get('confidence', 0) * 100
        
        summary = f"{signal_type} @ {entry:.2f} | Confidence: {confidence:.1f}%"
        
        if 'stop_loss' in signal and 'take_profit' in signal:
            sl = signal['stop_loss']
            tp = signal['take_profit']
            summary += f" | SL: {sl:.2f} | TP: {tp:.2f}"
        
        return summary
