"""
Trade execution and management.
"""
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from utils.logger import TradingLogger


class TradeExecutor:
    """Execute and manage trades."""
    
    def __init__(self, config: Dict[str, Any], risk_manager, database,
                 logger: Optional[TradingLogger] = None):
        """
        Initialize trade executor.
        
        Args:
            config: Configuration dictionary
            risk_manager: RiskManager instance
            database: TradingDatabase instance
            logger: Logger instance
        """
        self.config = config
        self.risk_manager = risk_manager
        self.database = database
        self.logger = logger or TradingLogger.get_logger(__name__)
        
        self.active_trades = []
    
    def execute_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute trading signal.
        
        Args:
            signal: Trading signal with SL/TP
            
        Returns:
            Trade dictionary or None if execution failed
        """
        # Check risk limits
        risk_check = self.risk_manager.check_risk_limits(self.active_trades)
        
        if not risk_check['can_trade']:
            self.logger.warning(f"Cannot execute trade: {risk_check}")
            return None
        
        # Create trade
        trade = {
            'trade_id': str(uuid.uuid4()),
            'signal_id': signal.get('id'),
            'entry_time': int(datetime.utcnow().timestamp()),
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'position_size': self.risk_manager.position_size,
            'direction': signal['signal_type'],
            'status': 'OPEN',
            'sl_distance': signal.get('sl_distance', 0),
            'tp_distance': signal.get('tp_distance', 0)
        }
        
        # In a real system, this would place the order via broker API
        # For now, we simulate execution
        
        try:
            # Save to database
            self.database.insert_trade(trade)
            
            # Add to active trades
            self.active_trades.append(trade)
            
            self.logger.info(f"Trade executed: {trade['direction']} @ {trade['entry_price']}")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Failed to execute trade: {e}")
            return None
    
    def update_trades(self, current_price: float) -> List[Dict[str, Any]]:
        """
        Update all active trades.
        
        Args:
            current_price: Current market price
            
        Returns:
            List of closed trades
        """
        closed_trades = []
        
        for trade in self.active_trades[:]:  # Copy list to allow removal
            # Check for exit
            exit_reason = self.risk_manager.check_trade_exit(trade, current_price)
            
            if exit_reason:
                closed_trade = self.close_trade(trade, current_price, exit_reason)
                if closed_trade:
                    closed_trades.append(closed_trade)
            else:
                # Check for trailing stop
                new_sl = self.risk_manager.should_trail_stop_loss(trade, current_price)
                if new_sl:
                    self.update_stop_loss(trade, new_sl)
        
        return closed_trades
    
    def close_trade(self, trade: Dict[str, Any], exit_price: float,
                    exit_reason: str) -> Optional[Dict[str, Any]]:
        """
        Close a trade.
        
        Args:
            trade: Trade to close
            exit_price: Exit price
            exit_reason: Reason for exit
            
        Returns:
            Closed trade dictionary
        """
        try:
            # Calculate P&L
            pnl_data = self.risk_manager.calculate_trade_pnl(trade, exit_price)
            
            # Update trade
            trade.update({
                'exit_time': int(datetime.utcnow().timestamp()),
                'exit_price': exit_price,
                'pnl': pnl_data['pnl'],
                'pnl_pips': pnl_data['pips'],
                'status': 'CLOSED',
                'exit_reason': exit_reason
            })
            
            # Update database
            self.database.update_trade(trade['trade_id'], {
                'exit_time': trade['exit_time'],
                'exit_price': exit_price,
                'pnl': pnl_data['pnl'],
                'pnl_pips': pnl_data['pips'],
                'status': 'CLOSED',
                'exit_reason': exit_reason
            })
            
            # Remove from active trades
            self.active_trades.remove(trade)
            
            # Update daily stats
            self.risk_manager.update_daily_stats(trade)
            
            result = "WIN" if pnl_data['pnl'] > 0 else "LOSS"
            self.logger.info(
                f"Trade closed: {result} | {trade['direction']} | "
                f"P&L: ${pnl_data['pnl']:.2f} ({pnl_data['pips']:.1f} pips) | "
                f"Reason: {exit_reason}"
            )
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error closing trade: {e}")
            return None
    
    def update_stop_loss(self, trade: Dict[str, Any], new_sl: float) -> None:
        """
        Update stop loss for a trade.
        
        Args:
            trade: Trade to update
            new_sl: New stop loss level
        """
        old_sl = trade['stop_loss']
        trade['stop_loss'] = new_sl
        
        # Update database
        self.database.update_trade(trade['trade_id'], {'stop_loss': new_sl})
        
        self.logger.info(f"Stop loss updated: {old_sl:.2f} -> {new_sl:.2f}")
    
    def close_all_trades(self, current_price: float, reason: str = "manual") -> List[Dict[str, Any]]:
        """
        Close all active trades.
        
        Args:
            current_price: Current market price
            reason: Reason for closing
            
        Returns:
            List of closed trades
        """
        closed_trades = []
        
        for trade in self.active_trades[:]:
            closed_trade = self.close_trade(trade, current_price, reason)
            if closed_trade:
                closed_trades.append(closed_trade)
        
        self.logger.info(f"Closed {len(closed_trades)} trades")
        
        return closed_trades
    
    def get_active_trades_summary(self) -> Dict[str, Any]:
        """
        Get summary of active trades.
        
        Returns:
            Summary dictionary
        """
        if not self.active_trades:
            return {
                'count': 0,
                'total_exposure': 0.0
            }
        
        total_exposure = sum(t['position_size'] for t in self.active_trades)
        
        return {
            'count': len(self.active_trades),
            'total_exposure': total_exposure,
            'trades': [
                {
                    'trade_id': t['trade_id'],
                    'direction': t['direction'],
                    'entry_price': t['entry_price'],
                    'stop_loss': t['stop_loss'],
                    'take_profit': t['take_profit']
                }
                for t in self.active_trades
            ]
        }
    
    def get_trade_duration(self, trade: Dict[str, Any]) -> int:
        """
        Get trade duration in minutes.
        
        Args:
            trade: Trade dictionary
            
        Returns:
            Duration in minutes
        """
        if 'exit_time' in trade:
            duration = trade['exit_time'] - trade['entry_time']
        else:
            duration = int(datetime.utcnow().timestamp()) - trade['entry_time']
        
        return duration // 60  # Convert to minutes
