"""
AllTick API wrapper for fetching GOLD market data.
"""
import requests
import time
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from utils.rate_limiter import RateLimiter
from utils.logger import TradingLogger


class AllTickDataFetcher:
    """Fetches historical and real-time data from AllTick API."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[TradingLogger] = None):
        """
        Initialize AllTick data fetcher.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.api_config = config['api']['alltick']
        self.base_url = self.api_config['base_url']
        self.token = self.api_config['token']
        self.logger = logger or TradingLogger.get_logger(__name__)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_calls=self.api_config['rate_limit_calls'],
            period=self.api_config['rate_limit_period']
        )
        
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, query_data: Dict[str, Any], 
                     max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Make API request with retry logic.
        
        Args:
            endpoint: API endpoint
            query_data: Query data to be JSON encoded
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response data or None on failure
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                
                # AllTick API format: wrap in trace + data structure
                request_payload = {
                    "trace": str(uuid.uuid4()),
                    "data": query_data
                }
                
                # AllTick API uses GET with token and query parameters
                params = {
                    'token': self.token,
                    'query': json.dumps(request_payload)
                }
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Log full response for debugging
                self.logger.debug(f"API Response: {json.dumps(data, indent=2)}")
                
                # AllTick uses 'ret' field, 200 = success
                if data.get('ret') == 200:
                    return data.get('data', {})
                else:
                    error_msg = data.get('msg', 'Unknown error')
                    self.logger.error(f"API error (ret {data.get('ret')}): {error_msg}")
                    self.logger.error(f"Request URL: {url}")
                    self.logger.error(f"Query data: {json.dumps(query_data, indent=2)}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Failed after {max_retries} attempts")
                    return None
        
        return None
    
    def fetch_historical_klines(self, symbol: str, interval: str,
                               start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Fetch historical kline data.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            interval: Timeframe (e.g., '5m')
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            List of candle dictionaries
        """
        self.logger.info(f"Fetching historical data for {symbol} from {start_time} to {end_time}")
        
        all_candles = []
        current_start = start_time
        
        # Convert interval to kline_type
        kline_type_map = {
            '1m': 1, '5m': 2, '15m': 3, '30m': 4,
            '1h': 5, '2h': 6, '4h': 7, '1d': 8, '1w': 9, '1M': 10
        }
        kline_type = kline_type_map.get(interval, 2)  # Default to 5m
        
        # Fetch in chunks (500 candles max per request)
        # For 5m candles: 500 candles = ~41 hours
        chunk_size = 500
        
        # Start from end and work backwards
        current_end = end_time
        
        while current_end > start_time and len(all_candles) < 100000:  # Safety limit
            query_data = {
                "code": symbol,
                "kline_type": kline_type,
                "kline_timestamp_end": int(current_end.timestamp()),
                "query_kline_num": chunk_size,
                "adjust_type": 0
            }
            
            data = self._make_request("kline", query_data)
            
            if data and 'kline_list' in data:
                candles = self._parse_klines(data['kline_list'])
                if candles:
                    # Filter candles within our date range
                    filtered = [c for c in candles if start_time.timestamp() <= c['timestamp'] <= end_time.timestamp()]
                    all_candles.extend(filtered)
                    self.logger.info(f"Fetched {len(filtered)} candles (total: {len(all_candles)})")
                    
                    # Update end time to oldest candle timestamp
                    oldest_ts = min(c['timestamp'] for c in candles)
                    current_end = datetime.fromtimestamp(oldest_ts, tz=timezone.utc)
                    
                    # If we got less than requested, we've reached the end
                    if len(candles) < chunk_size:
                        break
                else:
                    break
            else:
                self.logger.warning(f"No data received")
                break
            
            time.sleep(7)  # Respect rate limits (10 calls/min = 6 sec minimum)
        
        self.logger.info(f"Total candles fetched: {len(all_candles)}")
        return all_candles
    
    def fetch_latest_klines(self, symbol: str, interval: str, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch latest kline data.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            limit: Number of candles to fetch
            
        Returns:
            List of candle dictionaries
        """
        # Convert interval to kline_type
        kline_type_map = {
            '1m': 1, '5m': 2, '15m': 3, '30m': 4,
            '1h': 5, '2h': 6, '4h': 7, '1d': 8, '1w': 9, '1M': 10
        }
        kline_type = kline_type_map.get(interval, 2)
        
        query_data = {
            "code": symbol,
            "kline_type": kline_type,
            "kline_timestamp_end": 0,  # 0 = latest
            "query_kline_num": limit,
            "adjust_type": 0
        }
        
        data = self._make_request("kline", query_data)
        
        if data and 'kline_list' in data:
            candles = self._parse_klines(data['kline_list'])
            self.logger.debug(f"Fetched {len(candles)} latest candles")
            return candles
        
        return []
    
    def _parse_klines(self, klines: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Parse kline data from API response.
        
        Args:
            klines: List of kline dictionaries from AllTick API
            
        Returns:
            List of parsed candle dictionaries
        """
        candles = []
        
        for kline in klines:
            try:
                # AllTick kline format: dict with string values
                candle = {
                    'timestamp': int(kline['timestamp']),
                    'open': float(kline['open_price']),
                    'high': float(kline['high_price']),
                    'low': float(kline['low_price']),
                    'close': float(kline['close_price']),
                    'volume': float(kline.get('volume', 0))
                }
                candles.append(candle)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Failed to parse kline: {e}")
                continue
        
        return candles
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None
        """
        query_data = {
            "code": symbol
        }
        
        data = self._make_request("quote", query_data)
        
        if data and 'last_price' in data:
            return float(data['last_price'])
        
        return None
    
    def validate_connection(self) -> bool:
        """
        Validate API connection and credentials.
        
        Returns:
            True if connection is valid
        """
        try:
            symbol = self.config['trading']['symbol']
            candles = self.fetch_latest_klines(symbol, '5m', limit=1)
            
            if candles:
                self.logger.info("API connection validated successfully")
                return True
            else:
                self.logger.error("API connection validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"API connection validation error: {e}")
            return False
