"""
Telegram notifications for trading system.
"""
import asyncio
from telegram import Bot
from telegram.error import TelegramError
from typing import Dict, Any, Optional
from datetime import datetime
from utils.logger import TradingLogger


class TelegramNotifier:
    """Send trading notifications via Telegram."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[TradingLogger] = None):
        """
        Initialize Telegram notifier.
        
        Args:
            config: Telegram configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or TradingLogger.get_logger(__name__)
        
        telegram_config = config.get('telegram', {})
        
        self.enabled = telegram_config.get('enabled', True)
        self.bot_token = telegram_config.get('bot_token', '')
        self.chat_id = telegram_config.get('chat_id', '')
        self.templates = telegram_config.get('templates', {})
        self.notifications = telegram_config.get('notifications', {})
        
        if self.enabled and self.bot_token and self.chat_id:
            self.bot = Bot(token=self.bot_token)
        else:
            self.bot = None
            self.logger.warning("Telegram notifications disabled or not configured")
    
    async def _send_message(self, message: str) -> bool:
        """
        Send message via Telegram.
        
        Args:
            message: Message text
            
        Returns:
            True if sent successfully
        """
        if not self.bot:
            self.logger.debug(f"Telegram disabled, would send: {message}")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            return True
            
        except TelegramError as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_message_sync(self, message: str) -> bool:
        """
        Send message synchronously.
        
        Args:
            message: Message text
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a task
                asyncio.create_task(self._send_message(message))
                return True
            else:
                return loop.run_until_complete(self._send_message(message))
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_trade_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Send trade signal notification.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if sent successfully
        """
        if not self.notifications.get('trade_signals', True):
            return False
        
        signal_type = signal.get('signal_type', 'UNKNOWN')
        template_key = 'buy_signal' if signal_type == 'BUY' else 'sell_signal'
        template = self.templates.get(template_key, '')
        
        message = template.format(
            entry_price=signal.get('entry_price', 0),
            stop_loss=signal.get('stop_loss', 0),
            take_profit=signal.get('take_profit', 0),
            confidence=int(signal.get('confidence', 0) * 100)
        )
        
        return self.send_message_sync(message)
    
    def send_trade_outcome(self, trade: Dict[str, Any]) -> bool:
        """
        Send trade outcome notification.
        
        Args:
            trade: Closed trade dictionary
            
        Returns:
            True if sent successfully
        """
        if not self.notifications.get('trade_outcomes', True):
            return False
        
        pnl = trade.get('pnl', 0)
        pips = trade.get('pnl_pips', 0)
        
        if pnl > 0:
            template = self.templates.get('trade_win', '')
        else:
            template = self.templates.get('trade_loss', '')
            pips = abs(pips)
        
        # Calculate duration
        duration = 0
        if 'exit_time' in trade and 'entry_time' in trade:
            duration = (trade['exit_time'] - trade['entry_time']) // 60
        
        # Calculate R:R
        rr = 0
        if 'sl_distance' in trade and trade['sl_distance'] > 0:
            actual_move = abs(trade.get('exit_price', 0) - trade.get('entry_price', 0))
            rr = actual_move / trade['sl_distance']
        
        message = template.format(
            direction=trade.get('direction', 'UNKNOWN'),
            pips=abs(pips),
            rr=f"{rr:.1f}",
            duration=duration
        )
        
        return self.send_message_sync(message)
    
    def send_daily_summary(self, summary: Dict[str, Any]) -> bool:
        """
        Send daily trading summary.
        
        Args:
            summary: Daily summary dictionary
            
        Returns:
            True if sent successfully
        """
        if not self.notifications.get('daily_summary', True):
            return False
        
        template = self.templates.get('daily_summary', '')
        
        message = template.format(
            total_trades=summary.get('total_trades', 0),
            wins=summary.get('wins', 0),
            losses=summary.get('losses', 0),
            win_rate=int(summary.get('win_rate', 0)),
            net_pips=summary.get('net_pips', 0)
        )
        
        return self.send_message_sync(message)
    
    def send_system_alert(self, alert_message: str) -> bool:
        """
        Send system alert.
        
        Args:
            alert_message: Alert message
            
        Returns:
            True if sent successfully
        """
        if not self.notifications.get('system_alerts', True):
            return False
        
        template = self.templates.get('system_alert', '')
        message = template.format(message=alert_message)
        
        return self.send_message_sync(message)
    
    def send_error(self, error_message: str) -> bool:
        """
        Send error notification.
        
        Args:
            error_message: Error message
            
        Returns:
            True if sent successfully
        """
        if not self.notifications.get('errors', True):
            return False
        
        template = self.templates.get('error', '')
        message = template.format(error_message=error_message)
        
        return self.send_message_sync(message)
    
    def send_model_retrain(self, metrics: Dict[str, Any]) -> bool:
        """
        Send model retraining notification.
        
        Args:
            metrics: Training metrics
            
        Returns:
            True if sent successfully
        """
        if not self.notifications.get('model_retraining', True):
            return False
        
        template = self.templates.get('model_retrain', '')
        
        message = template.format(
            accuracy=int(metrics.get('accuracy', 0) * 100),
            precision=int(metrics.get('precision', 0) * 100),
            date=datetime.now().strftime('%Y-%m-%d %H:%M')
        )
        
        return self.send_message_sync(message)
    
    def send_custom_message(self, message: str) -> bool:
        """
        Send custom message.
        
        Args:
            message: Custom message text
            
        Returns:
            True if sent successfully
        """
        return self.send_message_sync(message)
