"""
Rate limiter for API calls to respect AllTick API limits.
"""
import time
from collections import deque
from threading import Lock
from typing import Callable, Any
import functools


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""
    
    def __init__(self, max_calls: int, period: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to rate limit function calls.
        
        Args:
            func: Function to rate limit
            
        Returns:
            Wrapped function with rate limiting
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with self.lock:
                now = time.time()
                
                # Remove calls outside the current window
                while self.calls and self.calls[0] <= now - self.period:
                    self.calls.popleft()
                
                # Check if we can make a call
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.period - (now - self.calls[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        now = time.time()
                        # Clean up again after sleeping
                        while self.calls and self.calls[0] <= now - self.period:
                            self.calls.popleft()
                
                # Record this call
                self.calls.append(now)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            
            # Remove calls outside the current window
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()
            
            # Check if we need to wait
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    while self.calls and self.calls[0] <= now - self.period:
                        self.calls.popleft()
            
            self.calls.append(now)
