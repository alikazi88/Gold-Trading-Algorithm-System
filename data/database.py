"""
SQLite database operations for the gold scalping system.
"""
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from contextlib import contextmanager
import os


class TradingDatabase:
    """Manages all database operations for the trading system."""
    
    def __init__(self, db_path: str):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._create_tables()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _create_tables(self) -> None:
        """Create all required database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Historical candles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL UNIQUE,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp)")
            
            # Feature engineering table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candle_id INTEGER NOT NULL,
                    timestamp INTEGER NOT NULL,
                    
                    -- Price Action Features
                    body_size REAL,
                    wick_size REAL,
                    candle_range REAL,
                    close_open_ratio REAL,
                    is_doji INTEGER,
                    is_engulfing INTEGER,
                    is_pin_bar INTEGER,
                    price_momentum_3 REAL,
                    price_momentum_5 REAL,
                    price_momentum_10 REAL,
                    distance_to_high_20 REAL,
                    distance_to_low_20 REAL,
                    
                    -- Support/Resistance Features
                    distance_to_support REAL,
                    distance_to_resistance REAL,
                    support_touches INTEGER,
                    resistance_touches INTEGER,
                    at_support INTEGER,
                    at_resistance INTEGER,
                    support_strength REAL,
                    resistance_strength REAL,
                    
                    -- Trend Features
                    higher_highs_count INTEGER,
                    higher_lows_count INTEGER,
                    lower_highs_count INTEGER,
                    lower_lows_count INTEGER,
                    trend_strength REAL,
                    trend_direction INTEGER,
                    trend_5m INTEGER,
                    trend_15m INTEGER,
                    trend_1h INTEGER,
                    
                    -- Smart Money Features
                    liquidity_sweep_bull INTEGER,
                    liquidity_sweep_bear INTEGER,
                    institutional_candle INTEGER,
                    volume_spike INTEGER,
                    session_asian INTEGER,
                    session_london INTEGER,
                    session_new_york INTEGER,
                    session_overlap INTEGER,
                    
                    -- Fair Value Gap Features
                    fvg_present INTEGER,
                    fvg_size REAL,
                    fvg_distance REAL,
                    fvg_age INTEGER,
                    fvg_bullish INTEGER,
                    fvg_bearish INTEGER,
                    
                    -- Order Block Features
                    ob_bull_present INTEGER,
                    ob_bear_present INTEGER,
                    ob_bull_distance REAL,
                    ob_bear_distance REAL,
                    ob_bull_strength REAL,
                    ob_bear_strength REAL,
                    ob_bull_mitigated INTEGER,
                    ob_bear_mitigated INTEGER,
                    
                    created_at INTEGER DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (candle_id) REFERENCES candles(id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_candle_id ON features(candle_id)")
            
            # Trade signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    confidence REAL NOT NULL,
                    features_snapshot TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            
            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    training_date INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    precision REAL NOT NULL,
                    recall REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    train_samples INTEGER,
                    test_samples INTEGER,
                    feature_importance TEXT,
                    hyperparameters TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Live trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    signal_id INTEGER,
                    entry_time INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    position_size REAL NOT NULL,
                    direction TEXT NOT NULL,
                    exit_time INTEGER,
                    exit_price REAL,
                    pnl REAL,
                    pnl_pips REAL,
                    status TEXT NOT NULL,
                    exit_reason TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
    
    def insert_candles(self, candles: List[Dict[str, Any]]) -> int:
        """
        Insert candles into database.
        
        Args:
            candles: List of candle dictionaries
            
        Returns:
            Number of candles inserted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
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
                        candle.get('volume', 0)
                    ))
                    if cursor.rowcount > 0:
                        inserted += 1
                except Exception as e:
                    print(f"Error inserting candle: {e}")
                    continue
            
            return inserted
    
    def get_candles(self, start_time: Optional[int] = None, 
                   end_time: Optional[int] = None, 
                   limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve candles from database.
        
        Args:
            start_time: Start timestamp (Unix)
            end_time: End timestamp (Unix)
            limit: Maximum number of candles
            
        Returns:
            DataFrame with candle data
        """
        with self.get_connection() as conn:
            query = "SELECT * FROM candles WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp ASC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            return df
    
    def insert_features(self, features: Dict[str, Any]) -> int:
        """
        Insert feature row into database.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Row ID of inserted features
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            columns = ', '.join(features.keys())
            placeholders = ', '.join(['?' for _ in features])
            
            cursor.execute(f"""
                INSERT INTO features ({columns})
                VALUES ({placeholders})
            """, tuple(features.values()))
            
            return cursor.lastrowid
    
    def get_features(self, start_time: Optional[int] = None,
                    end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve features from database.
        
        Args:
            start_time: Start timestamp (Unix)
            end_time: End timestamp (Unix)
            
        Returns:
            DataFrame with feature data
        """
        with self.get_connection() as conn:
            query = "SELECT * FROM features WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp ASC"
            
            df = pd.read_sql_query(query, conn, params=params)
            return df
    
    def insert_signal(self, signal: Dict[str, Any]) -> int:
        """
        Insert trade signal into database.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Signal ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO signals (timestamp, signal_type, entry_price, stop_loss, 
                                   take_profit, confidence, features_snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                signal['timestamp'],
                signal['signal_type'],
                signal['entry_price'],
                signal['stop_loss'],
                signal['take_profit'],
                signal['confidence'],
                signal.get('features_snapshot', '')
            ))
            
            return cursor.lastrowid
    
    def insert_trade(self, trade: Dict[str, Any]) -> int:
        """
        Insert trade into database.
        
        Args:
            trade: Trade dictionary
            
        Returns:
            Trade ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (trade_id, signal_id, entry_time, entry_price, 
                                  stop_loss, take_profit, position_size, direction, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['trade_id'],
                trade.get('signal_id'),
                trade['entry_time'],
                trade['entry_price'],
                trade['stop_loss'],
                trade['take_profit'],
                trade['position_size'],
                trade['direction'],
                trade['status']
            ))
            
            return cursor.lastrowid
    
    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> None:
        """
        Update trade in database.
        
        Args:
            trade_id: Trade ID
            updates: Dictionary of fields to update
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [trade_id]
            
            cursor.execute(f"""
                UPDATE trades SET {set_clause}
                WHERE trade_id = ?
            """, values)
    
    def insert_model_performance(self, performance: Dict[str, Any]) -> int:
        """
        Insert model performance metrics.
        
        Args:
            performance: Performance dictionary
            
        Returns:
            Performance record ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_performance (model_version, training_date, accuracy, 
                                             precision, recall, f1_score, train_samples,
                                             test_samples, feature_importance, hyperparameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                performance['model_version'],
                performance['training_date'],
                performance['accuracy'],
                performance['precision'],
                performance['recall'],
                performance['f1_score'],
                performance.get('train_samples', 0),
                performance.get('test_samples', 0),
                performance.get('feature_importance', ''),
                performance.get('hyperparameters', '')
            ))
            
            return cursor.lastrowid
    
    def get_latest_candle_timestamp(self) -> Optional[int]:
        """
        Get timestamp of the latest candle in database.
        
        Returns:
            Latest timestamp or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(timestamp) FROM candles")
            result = cursor.fetchone()
            return result[0] if result[0] else None
    
    def get_daily_trades(self, date: datetime) -> pd.DataFrame:
        """
        Get all trades for a specific date.
        
        Args:
            date: Date to query
            
        Returns:
            DataFrame with trade data
        """
        start_ts = int(date.replace(hour=0, minute=0, second=0).timestamp())
        end_ts = int(date.replace(hour=23, minute=59, second=59).timestamp())
        
        with self.get_connection() as conn:
            query = """
                SELECT * FROM trades 
                WHERE entry_time >= ? AND entry_time <= ?
                ORDER BY entry_time ASC
            """
            df = pd.read_sql_query(query, conn, params=(start_ts, end_ts))
            return df
