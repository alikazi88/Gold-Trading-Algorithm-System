"""
Structured logging utility for the gold scalping system.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional
import colorlog


class TradingLogger:
    """Centralized logging system with file and console handlers."""
    
    _loggers = {}
    
    @staticmethod
    def get_logger(name: str, log_file: Optional[str] = None, 
                   level: str = "INFO", max_bytes: int = 50*1024*1024,
                   backup_count: int = 5) -> logging.Logger:
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name (typically __name__)
            log_file: Path to log file
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_bytes: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            
        Returns:
            Configured logger instance
        """
        if name in TradingLogger._loggers:
            return TradingLogger._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = False
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler with colors
        console_handler = colorlog.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        TradingLogger._loggers[name] = logger
        return logger
    
    @staticmethod
    def setup_from_config(config: dict, name: str) -> logging.Logger:
        """
        Setup logger from configuration dictionary.
        
        Args:
            config: Configuration dictionary with logging settings
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        log_config = config.get('logging', {})
        return TradingLogger.get_logger(
            name=name,
            log_file=log_config.get('log_file', 'logs/trading_system.log'),
            level=log_config.get('level', 'INFO'),
            max_bytes=log_config.get('max_file_size_mb', 50) * 1024 * 1024,
            backup_count=log_config.get('backup_count', 5)
        )
