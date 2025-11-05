"""
Logging utilities for YOLOv3 realtime system
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name: str = None, log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logger with console and file output"""

    # Create logger
    logger = logging.getLogger(name or __name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file {log_file}: {e}")

    return logger

def get_logger(name: str = None) -> logging.Logger:
    """Get existing logger or create new one"""
    return logging.getLogger(name or __name__)

class PerformanceLogger:
    """Specialized logger for performance metrics"""

    def __init__(self, name: str = "performance", log_file: str = None):
        self.logger = setup_logger(name, log_file=log_file)
        self.start_time = None

    def start_timer(self, operation: str = "operation"):
        """Start timing an operation"""
        self.start_time = datetime.now()
        self.operation = operation

    def end_timer(self, additional_info: str = ""):
        """End timing and log the duration"""
        if self.start_time is None:
            self.logger.warning("Timer not started")
            return

        duration = (datetime.now() - self.start_time).total_seconds()
        info_str = f" - {additional_info}" if additional_info else ""
        self.logger.info(f"{self.operation} completed in {duration:.3f}s{info_str}")
        self.start_time = None

    def log_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log a performance metric"""
        unit_str = f" {unit}" if unit else ""
        self.logger.info(f"{metric_name}: {value:.3f}{unit_str}")

class SystemLogger:
    """System-wide logging manager"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.loggers = {}

    def get_logger(self, component: str, log_level: str = "INFO") -> logging.Logger:
        """Get logger for specific component"""
        if component not in self.loggers:
            log_file = self.log_dir / f"{component}.log"
            self.loggers[component] = setup_logger(
                f"yolov3.{component}",
                log_level=log_level,
                log_file=str(log_file)
            )
        return self.loggers[component]

    def set_global_level(self, log_level: str):
        """Set logging level for all loggers"""
        level = getattr(logging, log_level.upper(), logging.INFO)
        for logger in self.loggers.values():
            logger.setLevel(level)

    def cleanup_old_logs(self, days_to_keep: int = 7):
        """Clean up old log files"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)

            for log_file in self.log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    print(f"Deleted old log file: {log_file}")

        except Exception as e:
            print(f"Error cleaning up old logs: {e}")