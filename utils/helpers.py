"""
Common helper functions for the trading system.
"""
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def pips_to_price(pips: float, symbol: str = "XAUUSD") -> float:
    """
    Convert pips to price units.
    
    Args:
        pips: Number of pips
        symbol: Trading symbol
        
    Returns:
        Price in symbol units
    """
    if symbol == "XAUUSD":
        return pips * 0.1  # 1 pip = 0.1 for gold
    return pips * 0.0001  # Standard forex


def price_to_pips(price_diff: float, symbol: str = "XAUUSD") -> float:
    """
    Convert price difference to pips.
    
    Args:
        price_diff: Price difference
        symbol: Trading symbol
        
    Returns:
        Difference in pips
    """
    if symbol == "XAUUSD":
        return price_diff / 0.1
    return price_diff / 0.0001


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        ATR values
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLC data integrity.
    
    Args:
        df: DataFrame with OHLC columns
        
    Returns:
        Validated DataFrame with invalid rows removed
    """
    initial_len = len(df)
    
    # Check high >= low
    df = df[df['high'] >= df['low']]
    
    # Check high >= open, close
    df = df[(df['high'] >= df['open']) & (df['high'] >= df['close'])]
    
    # Check low <= open, close
    df = df[(df['low'] <= df['open']) & (df['low'] <= df['close'])]
    
    # Check for non-negative values
    df = df[(df['open'] > 0) & (df['high'] > 0) & 
            (df['low'] > 0) & (df['close'] > 0)]
    
    # Remove duplicates based on timestamp
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    
    removed = initial_len - len(df)
    if removed > 0:
        print(f"Removed {removed} invalid OHLC rows")
    
    return df


def get_session(timestamp: datetime) -> str:
    """
    Determine trading session from timestamp.
    
    Args:
        timestamp: UTC timestamp
        
    Returns:
        Session name (asian, london, new_york, or overlap)
    """
    hour = timestamp.hour
    
    # Asian: 00:00-08:00 UTC
    # London: 08:00-16:00 UTC
    # New York: 13:00-22:00 UTC
    # London/NY Overlap: 13:00-16:00 UTC
    
    if 13 <= hour < 16:
        return "overlap"
    elif 0 <= hour < 8:
        return "asian"
    elif 8 <= hour < 13:
        return "london"
    elif 13 <= hour < 22:
        return "new_york"
    else:
        return "asian"


def encode_session(session: str) -> Tuple[int, int, int, int]:
    """
    One-hot encode trading session.
    
    Args:
        session: Session name
        
    Returns:
        Tuple of (asian, london, new_york, overlap) binary values
    """
    encoding = {
        'asian': (1, 0, 0, 0),
        'london': (0, 1, 0, 0),
        'new_york': (0, 0, 1, 0),
        'overlap': (0, 0, 0, 1)
    }
    return encoding.get(session, (0, 0, 0, 0))


def is_round_number(price: float, threshold: float = 10.0) -> bool:
    """
    Check if price is near a psychological round number.
    
    Args:
        price: Price to check
        threshold: Distance threshold
        
    Returns:
        True if near round number
    """
    round_levels = [50, 100]
    for level in round_levels:
        if abs(price % level) < threshold or abs(price % level) > (level - threshold):
            return True
    return False


def calculate_position_size(account_balance: float, risk_percent: float,
                           stop_loss_pips: float, pip_value: float = 1.0) -> float:
    """
    Calculate position size based on risk management.
    
    Args:
        account_balance: Account balance
        risk_percent: Risk percentage per trade (e.g., 0.02 for 2%)
        stop_loss_pips: Stop loss in pips
        pip_value: Value of 1 pip for 1 lot
        
    Returns:
        Position size in lots
    """
    risk_amount = account_balance * risk_percent
    position_size = risk_amount / (stop_loss_pips * pip_value)
    return round(position_size, 2)


def format_timestamp(timestamp: datetime) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: Datetime object
        
    Returns:
        Formatted string
    """
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')


def get_date_range(days: int) -> Tuple[datetime, datetime]:
    """
    Get date range for data fetching.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date
