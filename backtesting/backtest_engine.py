"""
Backtesting engine for strategy validation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from utils.logger import TradingLogger
from utils.helpers import price_to_pips


class BacktestEngine:
    """Backtest trading strategy on historical data."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[TradingLogger] = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or TradingLogger.get_logger(__name__)
        
        backtest_config = config.get('backtest', {})
        
        self.initial_balance = backtest_config.get('initial_balance', 10000)
        self.commission = backtest_config.get('commission_per_trade', 0.0)
        self.slippage_pips = backtest_config.get('slippage_pips', 1.0)
        
        self.symbol = config.get('trading', {}).get('symbol', 'XAUUSD')
        
        # Results tracking
        self.trades = []
        self.equity_curve = []
        self.balance = self.initial_balance
    
    def run_backtest(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLC data
            signals_df: DataFrame with signals (timestamp, signal_type, entry_price, stop_loss, take_profit)
            
        Returns:
            Backtest results dictionary
        """
        self.logger.info(f"Starting backtest with {len(signals_df)} signals")
        
        # Reset state
        self.trades = []
        self.equity_curve = []
        self.balance = self.initial_balance
        
        # Merge signals with price data
        df = df.copy()
        df['signal'] = 0
        df['signal_sl'] = 0.0
        df['signal_tp'] = 0.0
        
        for _, signal in signals_df.iterrows():
            timestamp = signal['timestamp']
            mask = df['timestamp'] == timestamp
            if mask.any():
                df.loc[mask, 'signal'] = 1 if signal['signal_type'] == 'BUY' else -1
                df.loc[mask, 'signal_sl'] = signal['stop_loss']
                df.loc[mask, 'signal_tp'] = signal['take_profit']
        
        # Simulate trading
        active_trade = None
        
        for idx, row in df.iterrows():
            # Check for new signal
            if row['signal'] != 0 and active_trade is None:
                active_trade = self._open_trade(row)
            
            # Update active trade
            if active_trade:
                exit_result = self._check_trade_exit(active_trade, row)
                if exit_result:
                    self._close_trade(active_trade, exit_result, row)
                    active_trade = None
            
            # Record equity
            current_equity = self.balance
            if active_trade:
                unrealized_pnl = self._calculate_unrealized_pnl(active_trade, row['close'])
                current_equity += unrealized_pnl
            
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'equity': current_equity
            })
        
        # Close any remaining trade
        if active_trade:
            last_row = df.iloc[-1]
            self._close_trade(active_trade, {'reason': 'end_of_data', 'price': last_row['close']}, last_row)
        
        # Calculate results
        results = self._calculate_results()
        
        self.logger.info(f"Backtest completed: {len(self.trades)} trades executed")
        
        return results
    
    def _open_trade(self, row: pd.Series) -> Dict[str, Any]:
        """Open a new trade."""
        direction = 'BUY' if row['signal'] == 1 else 'SELL'
        entry_price = row['close']
        
        # Apply slippage
        slippage = self.slippage_pips * 0.1  # Convert to price
        if direction == 'BUY':
            entry_price += slippage
        else:
            entry_price -= slippage
        
        trade = {
            'entry_time': row['timestamp'],
            'entry_price': entry_price,
            'direction': direction,
            'stop_loss': row['signal_sl'],
            'take_profit': row['signal_tp'],
            'position_size': 0.1,  # Fixed for backtest
            'status': 'OPEN'
        }
        
        return trade
    
    def _check_trade_exit(self, trade: Dict[str, Any], row: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if trade should be exited."""
        direction = trade['direction']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        
        if direction == 'BUY':
            if row['low'] <= stop_loss:
                return {'reason': 'stop_loss', 'price': stop_loss}
            elif row['high'] >= take_profit:
                return {'reason': 'take_profit', 'price': take_profit}
        else:  # SELL
            if row['high'] >= stop_loss:
                return {'reason': 'stop_loss', 'price': stop_loss}
            elif row['low'] <= take_profit:
                return {'reason': 'take_profit', 'price': take_profit}
        
        return None
    
    def _close_trade(self, trade: Dict[str, Any], exit_result: Dict[str, Any], row: pd.Series) -> None:
        """Close a trade and update balance."""
        exit_price = exit_result['price']
        exit_reason = exit_result['reason']
        
        # Calculate P&L
        if trade['direction'] == 'BUY':
            price_diff = exit_price - trade['entry_price']
        else:
            price_diff = trade['entry_price'] - exit_price
        
        pips = price_to_pips(price_diff, self.symbol)
        
        # Calculate monetary P&L (assuming $10 per pip per lot)
        pip_value = 10.0 * trade['position_size']
        pnl = pips * pip_value - self.commission
        
        # Update balance
        self.balance += pnl
        
        # Record trade
        trade.update({
            'exit_time': row['timestamp'],
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pips': pips,
            'status': 'CLOSED'
        })
        
        self.trades.append(trade)
    
    def _calculate_unrealized_pnl(self, trade: Dict[str, Any], current_price: float) -> float:
        """Calculate unrealized P&L for open trade."""
        if trade['direction'] == 'BUY':
            price_diff = current_price - trade['entry_price']
        else:
            price_diff = trade['entry_price'] - current_price
        
        pips = price_to_pips(price_diff, self.symbol)
        pip_value = 10.0 * trade['position_size']
        
        return pips * pip_value
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest results."""
        if not self.trades:
            return self._get_empty_results()
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        total_pips = trades_df['pips'].sum()
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        profit_factor = (trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                        abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())) if losing_trades > 0 else 0
        
        # Drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
        equity_df['drawdown_pct'] = (equity_df['drawdown'] / equity_df['peak']) * 100
        
        max_drawdown = equity_df['drawdown'].min()
        max_drawdown_pct = equity_df['drawdown_pct'].min()
        
        # Return
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        # Trade duration
        trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']) / 60  # minutes
        avg_duration = trades_df['duration'].mean()
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pips': total_pips,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'total_return': total_return,
            'final_balance': self.balance,
            'avg_trade_duration_minutes': avg_duration,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
        
        return results
    
    def _get_empty_results(self) -> Dict[str, Any]:
        """Get empty results structure."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'total_pips': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'total_return': 0.0,
            'final_balance': self.initial_balance,
            'avg_trade_duration_minutes': 0.0,
            'trades': [],
            'equity_curve': []
        }
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print backtest results."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"\nInitial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${results['final_balance']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"\nTotal Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"\nTotal P&L: ${results['total_pnl']:,.2f}")
        print(f"Total Pips: {results['total_pips']:.1f}")
        print(f"Average Win: ${results['avg_win']:.2f}")
        print(f"Average Loss: ${results['avg_loss']:.2f}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"\nMax Drawdown: ${results['max_drawdown']:,.2f} ({results['max_drawdown_pct']:.2f}%)")
        print(f"Avg Trade Duration: {results['avg_trade_duration_minutes']:.1f} minutes")
        print("="*60 + "\n")
