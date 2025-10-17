#!/usr/bin/env python3
"""
Generate mock GOLD trading data for testing the system.
This creates realistic OHLC data without needing API access.
"""
import sqlite3
import random
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import TradingLogger

class MockDataGenerator:
    """Generate realistic mock GOLD price data."""
    
    def __init__(self, db_path: str = "data/gold_trading.db"):
        self.db_path = db_path
        self.logger = TradingLogger.get_logger(__name__)
        
    def generate_realistic_candles(self, days: int = 365) -> list:
        """
        Generate realistic GOLD price candles.
        
        Args:
            days: Number of days of data to generate
            
        Returns:
            List of candle dictionaries
        """
        self.logger.info(f"Generating {days} days of mock GOLD data...")
        
        # GOLD typical characteristics
        base_price = 2650.0  # Current approximate GOLD price
        daily_volatility = 0.015  # 1.5% daily volatility
        trend_strength = 0.0002  # Slight upward bias
        
        # Calculate number of 5-minute candles
        candles_per_day = 288  # 24 hours * 60 minutes / 5 minutes
        total_candles = days * candles_per_day
        
        # Start time
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        current_time = start_time
        
        candles = []
        current_price = base_price
        
        for i in range(total_candles):
            timestamp = int(current_time.timestamp())
            
            # Add some trend and mean reversion
            trend = trend_strength * (random.random() - 0.4)  # Slight upward bias
            mean_reversion = (base_price - current_price) * 0.001
            
            # Calculate price movement
            price_change = (trend + mean_reversion + random.gauss(0, daily_volatility / 48))
            
            # Generate OHLC
            open_price = current_price
            
            # Intrabar volatility
            high_move = abs(random.gauss(0, 0.0005)) * open_price
            low_move = abs(random.gauss(0, 0.0005)) * open_price
            
            high_price = open_price + high_move
            low_price = open_price - low_move
            
            # Close price
            close_price = open_price + (price_change * open_price)
            close_price = max(low_price, min(high_price, close_price))
            
            # Ensure high/low are correct
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Volume (higher during certain hours)
            hour = current_time.hour
            volume_multiplier = 1.0
            if 8 <= hour <= 10 or 13 <= hour <= 15:  # London/NY open
                volume_multiplier = 2.0
            volume = random.uniform(100, 500) * volume_multiplier
            
            candles.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 2)
            })
            
            # Update for next candle
            current_price = close_price
            current_time += timedelta(minutes=5)
            
            # Progress indicator
            if (i + 1) % 10000 == 0:
                progress = (i + 1) / total_candles * 100
                self.logger.info(f"Generated {i + 1}/{total_candles} candles ({progress:.1f}%)")
        
        self.logger.info(f"✅ Generated {len(candles)} candles")
        return candles
    
    def insert_candles(self, candles: list):
        """Insert candles into database."""
        self.logger.info("Inserting candles into database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert candles
        inserted = 0
        for candle in candles:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO candles (timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    candle['timestamp'],
                    candle['open'],
                    candle['high'],
                    candle['low'],
                    candle['close'],
                    candle['volume']
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                self.logger.error(f"Error inserting candle: {e}")
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"✅ Inserted {inserted} new candles into database")
        return inserted
    
    def get_stats(self):
        """Get statistics about the generated data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM candles")
        total_candles = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM candles")
        min_ts, max_ts = cursor.fetchone()
        
        if min_ts and max_ts:
            start_date = datetime.fromtimestamp(min_ts, tz=timezone.utc)
            end_date = datetime.fromtimestamp(max_ts, tz=timezone.utc)
            days = (end_date - start_date).days
        else:
            start_date = end_date = None
            days = 0
        
        cursor.execute("SELECT MIN(low), MAX(high), AVG(close) FROM candles")
        min_price, max_price, avg_price = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_candles': total_candles,
            'start_date': start_date,
            'end_date': end_date,
            'days': days,
            'min_price': min_price,
            'max_price': max_price,
            'avg_price': avg_price
        }

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate mock GOLD trading data')
    parser.add_argument('--days', type=int, default=365, help='Number of days to generate (default: 365)')
    parser.add_argument('--db', type=str, default='data/gold_trading.db', help='Database path')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MOCK GOLD DATA GENERATOR")
    print("=" * 80)
    print(f"\nGenerating {args.days} days of realistic GOLD price data...")
    print("This will create 5-minute candles with realistic price action.\n")
    
    generator = MockDataGenerator(args.db)
    
    # Generate candles
    candles = generator.generate_realistic_candles(args.days)
    
    # Insert into database
    inserted = generator.insert_candles(candles)
    
    # Show statistics
    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)
    
    stats = generator.get_stats()
    print(f"\nTotal Candles: {stats['total_candles']:,}")
    print(f"Date Range: {stats['start_date']} to {stats['end_date']}")
    print(f"Days of Data: {stats['days']}")
    print(f"Price Range: ${stats['min_price']:.2f} - ${stats['max_price']:.2f}")
    print(f"Average Price: ${stats['avg_price']:.2f}")
    
    print("\n" + "=" * 80)
    print("✅ MOCK DATA GENERATION COMPLETE")
    print("=" * 80)
    print("\nYou can now:")
    print("1. Train the model: python train_model.py")
    print("2. Run backtests: python example_usage.py")
    print("3. Test the system without API access\n")

if __name__ == "__main__":
    main()
