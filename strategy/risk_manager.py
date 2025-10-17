"""
Risk management for gold scalping system.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from utils.logger import TradingLogger
from utils.helpers import price_to_pips, pips_to_price


class RiskManager:
    """Manage trading risk and position sizing."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[TradingLogger] = None):
        """
        Initialize risk manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or TradingLogger.get_logger(__name__)
        
        trading_config = config.get('trading', {})
        risk_config = config.get('risk', {})
        
        self.position_size = trading_config.get('position_size', 0.1)
        self.max_concurrent_trades = trading_config.get('max_concurrent_trades', 1)
        self.max_trades_per_day = trading_config.get('max_trades_per_day', 15)
        self.daily_drawdown_limit = trading_config.get('daily_drawdown_limit', 0.03)
        
        self.sl_atr_multiplier = risk_config.get('stop_loss_atr_multiplier', 1.0)
        self.tp_ratio = risk_config.get('take_profit_ratio', 2.0)
        self.trail_to_be_at_rr = risk_config.get('trail_to_breakeven_at_rr', 1.0)
        self.atr_period = risk_config.get('atr_period', 20)
        
        self.symbol = trading_config.get('symbol', 'XAUUSD')
        
        # Track daily stats
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.current_date = datetime.utcnow().date()
    
    def calculate_stop_loss_take_profit(self, signal: Dict[str, Any], 
                                        atr: float) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            signal: Trading signal
            atr: Current ATR value
            
        Returns:
            Dictionary with SL and TP levels
        """
        entry_price = signal['entry_price']
        signal_type = signal['signal_type']
        
        # Calculate stop loss distance
        sl_distance = atr * self.sl_atr_multiplier
        tp_distance = sl_distance * self.tp_ratio
        
        if signal_type == 'BUY':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SELL
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        # Convert to pips for logging
        sl_pips = price_to_pips(sl_distance, self.symbol)
        tp_pips = price_to_pips(tp_distance, self.symbol)
        
        self.logger.info(f"SL: {stop_loss:.2f} ({sl_pips:.1f} pips), TP: {take_profit:.2f} ({tp_pips:.1f} pips)")
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sl_distance': sl_distance,
            'tp_distance': tp_distance,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'risk_reward_ratio': self.tp_ratio
        }
    
    def calculate_position_size(self, account_balance: float, 
                                risk_percent: float,
                                sl_pips: float) -> float:
        """
        Calculate position size based on risk.
        
        Args:
            account_balance: Account balance
            risk_percent: Risk percentage per trade
            sl_pips: Stop loss in pips
            
        Returns:
            Position size in lots
        """
        # For gold, 1 pip = $0.10 per micro lot (0.01)
        # For 1 standard lot, 1 pip = $10
        pip_value_per_lot = 10.0
        
        risk_amount = account_balance * risk_percent
        position_size = risk_amount / (sl_pips * pip_value_per_lot)
        
        # Round to 2 decimal places
        position_size = round(position_size, 2)
        
        # Use configured position size if dynamic sizing not used
        return self.position_size
    
    def check_risk_limits(self, active_trades: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Check if risk limits are respected.
        
        Args:
            active_trades: List of active trades
            
        Returns:
            Dictionary with limit check results
        """
        # Reset daily stats if new day
        current_date = datetime.utcnow().date()
        if current_date != self.current_date:
            self.daily_trades = []
            self.daily_pnl = 0.0
            self.current_date = current_date
        
        # Check concurrent trades
        concurrent_ok = len(active_trades) < self.max_concurrent_trades
        
        # Check daily trade limit
        daily_trades_ok = len(self.daily_trades) < self.max_trades_per_day
        
        # Check daily drawdown
        drawdown_ok = True
        if self.daily_pnl < 0:
            # Calculate drawdown percentage (would need account balance)
            # For now, just check absolute loss
            drawdown_ok = abs(self.daily_pnl) < 1000  # Placeholder
        
        return {
            'concurrent_trades_ok': concurrent_ok,
            'daily_trades_ok': daily_trades_ok,
            'drawdown_ok': drawdown_ok,
            'can_trade': concurrent_ok and daily_trades_ok and drawdown_ok
        }
    
    def should_trail_stop_loss(self, trade: Dict[str, Any], 
                               current_price: float) -> Optional[float]:
        """
        Determine if stop loss should be trailed to breakeven.
        
        Args:
            trade: Trade dictionary
            current_price: Current market price
            
        Returns:
            New stop loss level or None
        """
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        direction = trade['direction']
        
        # Calculate current profit in terms of risk
        if direction == 'BUY':
            profit = current_price - entry_price
            risk = entry_price - stop_loss
        else:  # SELL
            profit = entry_price - current_price
            risk = stop_loss - entry_price
        
        if risk == 0:
            return None
        
        current_rr = profit / risk
        
        # Trail to breakeven if target RR reached
        if current_rr >= self.trail_to_be_at_rr:
            if direction == 'BUY':
                if stop_loss < entry_price:
                    self.logger.info(f"Trailing SL to breakeven for BUY trade")
                    return entry_price
            else:  # SELL
                if stop_loss > entry_price:
                    self.logger.info(f"Trailing SL to breakeven for SELL trade")
                    return entry_price
        
        return None
    
    def check_trade_exit(self, trade: Dict[str, Any], 
                        current_price: float) -> Optional[str]:
        """
        Check if trade should be exited.
        
        Args:
            trade: Trade dictionary
            current_price: Current market price
            
        Returns:
            Exit reason or None
        """
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        direction = trade['direction']
        
        if direction == 'BUY':
            if current_price <= stop_loss:
                return 'stop_loss'
            elif current_price >= take_profit:
                return 'take_profit'
        else:  # SELL
            if current_price >= stop_loss:
                return 'stop_loss'
            elif current_price <= take_profit:
                return 'take_profit'
        
        return None
    
    def calculate_trade_pnl(self, trade: Dict[str, Any], 
                           exit_price: float) -> Dict[str, float]:
        """
        Calculate trade P&L.
        
        Args:
            trade: Trade dictionary
            exit_price: Exit price
            
        Returns:
            Dictionary with P&L metrics
        """
        entry_price = trade['entry_price']
        direction = trade['direction']
        position_size = trade.get('position_size', self.position_size)
        
        # Calculate price difference
        if direction == 'BUY':
            price_diff = exit_price - entry_price
        else:  # SELL
            price_diff = entry_price - exit_price
        
        # Convert to pips
        pips = price_to_pips(price_diff, self.symbol)
        
        # Calculate P&L in currency
        # For gold: 1 pip = $10 per standard lot
        pip_value = 10.0 * position_size
        pnl = pips * pip_value
        
        # Calculate R-multiple
        risk = trade.get('sl_distance', 0)
        if risk > 0:
            r_multiple = abs(price_diff) / risk
            if price_diff < 0:
                r_multiple = -r_multiple
        else:
            r_multiple = 0
        
        return {
            'pnl': pnl,
            'pips': pips,
            'r_multiple': r_multiple,
            'exit_price': exit_price
        }
    
    def update_daily_stats(self, trade_result: Dict[str, Any]) -> None:
        """
        Update daily trading statistics.
        
        Args:
            trade_result: Trade result dictionary
        """
        self.daily_trades.append(trade_result)
        self.daily_pnl += trade_result.get('pnl', 0)
        
        self.logger.info(f"Daily stats: {len(self.daily_trades)} trades, PnL: ${self.daily_pnl:.2f}")
    
    def get_daily_summary(self) -> Dict[str, Any]:
        """
        Get daily trading summary.
        
        Returns:
            Dictionary with daily statistics
        """
        if not self.daily_trades:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'net_pips': 0.0,
                'net_pnl': 0.0
            }
        
        wins = sum(1 for t in self.daily_trades if t.get('pnl', 0) > 0)
        losses = len(self.daily_trades) - wins
        win_rate = (wins / len(self.daily_trades)) * 100
        
        net_pips = sum(t.get('pips', 0) for t in self.daily_trades)
        
        return {
            'total_trades': len(self.daily_trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'net_pips': net_pips,
            'net_pnl': self.daily_pnl
        }
