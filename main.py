"""
Main orchestrator for the Gold Scalping Trading System.
"""
import sys
import os
import time
import signal
from datetime import datetime, timedelta, timezone
from typing import Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import load_config, calculate_atr
from utils.logger import TradingLogger
from data.database import TradingDatabase
from data.fetcher import AllTickDataFetcher
from data.preprocessor import DataPreprocessor
from models.feature_engineering import FeatureEngineer
from models.random_forest_model import GoldScalpingModel
from strategy.signal_generator import SignalGenerator
from strategy.risk_manager import RiskManager
from strategy.trade_executor import TradeExecutor
from notifications.telegram_bot import TelegramNotifier


class GoldScalpingSystem:
    """Main trading system orchestrator."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize trading system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        telegram_config = load_config("config/telegram_config.yaml")
        self.config.update(telegram_config)
        
        # Setup logger
        self.logger = TradingLogger.setup_from_config(self.config, __name__)
        
        # Initialize components
        self.database = TradingDatabase(self.config['database']['path'])
        self.data_fetcher = AllTickDataFetcher(self.config, self.logger)
        self.preprocessor = DataPreprocessor(self.logger)
        self.feature_engineer = FeatureEngineer(self.config, self.logger)
        self.risk_manager = RiskManager(self.config, self.logger)
        self.telegram = TelegramNotifier(self.config, self.logger)
        
        # Load trained model
        self.model = None
        self.signal_generator = None
        self.trade_executor = None
        
        # System state
        self.running = False
        self.last_update = None
        
    def initialize(self) -> bool:
        """
        Initialize the trading system.
        
        Returns:
            True if initialization successful
        """
        self.logger.info("="*60)
        self.logger.info("GOLD SCALPING SYSTEM INITIALIZATION")
        self.logger.info("="*60)
        
        try:
            # Validate API connection
            self.logger.info("Validating API connection...")
            if not self.data_fetcher.validate_connection():
                self.logger.error("API connection validation failed")
                return False
            
            # Load trained model
            self.logger.info("Loading trained model...")
            if not self._load_latest_model():
                self.logger.error("Failed to load model")
                return False
            
            # Initialize signal generator
            self.signal_generator = SignalGenerator(self.model, self.config, self.logger)
            
            # Initialize trade executor
            self.trade_executor = TradeExecutor(
                self.config, self.risk_manager, self.database, self.logger
            )
            
            # Send startup notification
            self.telegram.send_system_alert("Gold Scalping System started successfully")
            
            self.logger.info("System initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}", exc_info=True)
            self.telegram.send_error(f"System initialization failed: {e}")
            return False
    
    def _load_latest_model(self) -> bool:
        """Load the latest trained model."""
        model_dir = "models/saved"
        
        if not os.path.exists(model_dir):
            self.logger.error(f"Model directory not found: {model_dir}")
            return False
        
        # Find latest model file
        model_files = [f for f in os.listdir(model_dir) if f.startswith("gold_scalping_model_") and f.endswith(".pkl")]
        
        if not model_files:
            self.logger.error("No trained model found")
            return False
        
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(model_dir, latest_model)
        
        # Load metadata
        version = latest_model.replace("gold_scalping_model_", "").replace(".pkl", "")
        metadata_path = os.path.join(model_dir, f"model_metadata_{version}.json")
        
        self.model = GoldScalpingModel(self.config, self.logger)
        self.model.load_model(model_path, metadata_path if os.path.exists(metadata_path) else None)
        
        self.logger.info(f"Loaded model: {latest_model}")
        return True
    
    def fetch_initial_data(self, days: int = 365) -> bool:
        """
        Fetch initial historical data.
        
        Args:
            days: Number of days to fetch
            
        Returns:
            True if successful
        """
        self.logger.info(f"Fetching {days} days of historical data...")
        
        try:
            symbol = self.config['trading']['symbol']
            interval = self.config['trading']['timeframe']
            
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)
            
            candles = self.data_fetcher.fetch_historical_klines(
                symbol, interval, start_time, end_time
            )
            
            if not candles:
                self.logger.error("No data fetched")
                return False
            
            # Insert into database
            inserted = self.database.insert_candles(candles)
            self.logger.info(f"Inserted {inserted} candles into database")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching initial data: {e}", exc_info=True)
            return False
    
    def update_data(self) -> bool:
        """
        Update with latest candles.
        
        Returns:
            True if successful
        """
        try:
            symbol = self.config['trading']['symbol']
            interval = self.config['trading']['timeframe']
            
            # Fetch latest candles
            candles = self.data_fetcher.fetch_latest_klines(symbol, interval, limit=100)
            
            if candles:
                inserted = self.database.insert_candles(candles)
                if inserted > 0:
                    self.logger.info(f"📊 Updated with {inserted} new candles")
                else:
                    self.logger.info("✓ Data up to date (no new candles)")
                return True
            else:
                self.logger.warning("⚠️  No candles received (rate limit or API issue)")
                return False
            
        except Exception as e:
            self.logger.error(f"Error updating data: {e}")
            return False
    
    def process_latest_candle(self) -> None:
        """Process the latest candle and generate signals."""
        try:
            # Get recent candles
            df = self.database.get_candles(limit=200)
            
            if len(df) < 50:
                self.logger.warning("Insufficient data for processing")
                return
            
            # Preprocess
            df = self.preprocessor.clean_candle_data(df)
            
            # Calculate features
            df = self.feature_engineer.calculate_all_features(df)
            
            # Get latest candle
            latest_idx = len(df) - 1
            latest_candle = df.iloc[latest_idx]
            current_price = latest_candle['close']
            
            # Calculate ATR
            atr_values = calculate_atr(
                df['high'].values, df['low'].values, df['close'].values,
                period=self.config['risk']['atr_period']
            )
            current_atr = atr_values[latest_idx]
            
            # Extract features
            features = self.feature_engineer.calculate_features_for_candle(df, latest_idx)
            
            if not features:
                self.logger.warning("Failed to extract features")
                return
            
            # Generate signal
            signal = self.signal_generator.generate_signal(features, current_price, current_atr)
            
            if signal:
                signal_type = signal['signal']
                confidence = signal['confidence'] * 100
                self.logger.info(f"🎯 Signal Generated: {signal_type} (Confidence: {confidence:.1f}%)")
                
                # Validate signal
                market_context = {'current_price': current_price, 'atr': current_atr}
                
                if self.signal_generator.validate_signal(signal, market_context):
                    # Calculate SL/TP
                    sl_tp = self.risk_manager.calculate_stop_loss_take_profit(signal, current_atr)
                    signal.update(sl_tp)
                    
                    self.logger.info(f"   Entry: ${current_price:.2f} | SL: ${signal['stop_loss']:.2f} | TP: ${signal['take_profit']:.2f}")
                    
                    # Enhance with context
                    signal = self.signal_generator.enhance_signal_with_context(signal, df)
                    
                    # Save signal to database
                    signal_id = self.database.insert_signal(signal)
                    signal['id'] = signal_id
                    
                    # Send notification
                    self.telegram.send_trade_signal(signal)
                    
                    # Execute trade
                    trade = self.trade_executor.execute_signal(signal)
                    
                    if trade:
                        self.logger.info(f"✅ Trade Executed: {signal_type} at ${current_price:.2f}")
                    else:
                        self.logger.warning("⚠️  Trade execution failed (risk limits or validation)")
                else:
                    self.logger.info(f"⏭️  Signal rejected by validation")
            else:
                self.logger.info("➖ No trade signal (NO_TRADE)")
            
            # Update active trades
            closed_trades = self.trade_executor.update_trades(current_price)
            
            # Log active trades status
            active_count = len(self.trade_executor.active_trades)
            if active_count > 0:
                self.logger.info(f"📈 Active Trades: {active_count}")
            
            # Send notifications for closed trades
            for trade in closed_trades:
                pnl = trade.get('pnl', 0)
                outcome = trade.get('outcome', 'unknown')
                pnl_symbol = "🟢" if pnl > 0 else "🔴"
                self.logger.info(f"{pnl_symbol} Trade Closed: {outcome.upper()} | P&L: ${pnl:.2f}")
                self.telegram.send_trade_outcome(trade)
            
        except Exception as e:
            self.logger.error(f"Error processing candle: {e}", exc_info=True)
            self.telegram.send_error(f"Processing error: {e}")
    
    def run(self) -> None:
        """Run the trading system main loop."""
        self.logger.info("="*60)
        self.logger.info("🚀 STARTING LIVE TRADING SYSTEM")
        self.logger.info("="*60)
        self.logger.info(f"Symbol: {self.config['trading']['symbol']}")
        self.logger.info(f"Timeframe: {self.config['trading']['timeframe']}")
        self.logger.info(f"Update Interval: Every {self.config['data']['update_interval_seconds']} seconds")
        self.logger.info(f"Model: {self.model.model_path if hasattr(self.model, 'model_path') else 'Loaded'}")
        self.logger.info("="*60)
        
        self.running = True
        update_interval = self.config['data']['update_interval_seconds']
        cycle_count = 0
        
        try:
            while self.running:
                cycle_count += 1
                current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                
                self.logger.info("")
                self.logger.info(f"{'='*60}")
                self.logger.info(f"⏰ Cycle #{cycle_count} | {current_time}")
                self.logger.info(f"{'='*60}")
                
                # Update data
                data_updated = self.update_data()
                
                if data_updated:
                    # Process latest candle
                    self.process_latest_candle()
                else:
                    self.logger.warning("⚠️  Skipping processing due to data update failure")
                
                # Show daily stats
                daily_summary = self.risk_manager.get_daily_summary()
                if daily_summary.get('total_trades', 0) > 0:
                    self.logger.info(f"📊 Today: {daily_summary['total_trades']} trades | "
                                   f"P&L: ${daily_summary.get('total_pnl', 0):.2f} | "
                                   f"Win Rate: {daily_summary.get('win_rate', 0):.1f}%")
                
                # Wait for next update
                self.logger.info(f"⏳ Waiting {update_interval} seconds until next cycle...")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}", exc_info=True)
            self.telegram.send_error(f"System crashed: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self) -> None:
        """Shutdown the trading system gracefully."""
        self.logger.info("Shutting down trading system...")
        
        self.running = False
        
        # Close all active trades
        if self.trade_executor:
            df = self.database.get_candles(limit=1)
            if len(df) > 0:
                current_price = df.iloc[-1]['close']
                self.trade_executor.close_all_trades(current_price, "system_shutdown")
        
        # Send daily summary
        daily_summary = self.risk_manager.get_daily_summary()
        self.telegram.send_daily_summary(daily_summary)
        
        # Send shutdown notification
        self.telegram.send_system_alert("Gold Scalping System stopped")
        
        self.logger.info("System shutdown complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gold Scalping Trading System")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--fetch-data', action='store_true',
                       help='Fetch initial historical data')
    parser.add_argument('--days', type=int, default=365,
                       help='Number of days to fetch (with --fetch-data)')
    
    args = parser.parse_args()
    
    # Create system
    system = GoldScalpingSystem(args.config)
    
    # Fetch initial data if requested
    if args.fetch_data:
        print("Fetching historical data...")
        if system.fetch_initial_data(args.days):
            print("Data fetch completed successfully")
        else:
            print("Data fetch failed")
        return
    
    # Initialize and run
    if system.initialize():
        # Setup signal handlers
        def signal_handler(sig, frame):
            system.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run system
        system.run()
    else:
        print("System initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
