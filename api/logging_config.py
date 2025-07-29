import logging
import logging.config
import json
import os
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'pod_id'):
            log_entry['pod_id'] = record.pod_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'endpoint'):
            log_entry['endpoint'] = record.endpoint
        if hasattr(record, 'execution_time'):
            log_entry['execution_time_ms'] = record.execution_time
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

def setup_logging(
    level: str = None,
    json_format: bool = None,
    log_file: str = None
) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to use JSON formatting
        log_file: Optional log file path
    """
    # Get configuration from environment variables with defaults
    level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    json_format = json_format if json_format is not None else os.getenv("LOG_JSON", "true").lower() == "true"
    log_file = log_file or os.getenv("LOG_FILE")
    
    # Choose formatter
    if json_format:
        formatter_class = JSONFormatter
        format_string = None  # JSONFormatter doesn't use format string
    else:
        formatter_class = logging.Formatter
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup handlers
    handlers = ["console"]
    handler_config = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "default",
            "stream": "ext://sys.stdout"
        }
    }
    
    # Add file handler if specified
    if log_file:
        handlers.append("file")
        handler_config["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "default",
            "filename": log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        }
    
    # Logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": formatter_class,
            }
        },
        "handlers": handler_config,
        "root": {
            "level": level,
            "handlers": handlers
        },
        "loggers": {
            # Suppress noisy third-party loggers
            "urllib3.connectionpool": {
                "level": "WARNING",
                "propagate": True
            },
            "requests.packages.urllib3": {
                "level": "WARNING", 
                "propagate": True
            },
            "httpcore": {
                "level": "WARNING",
                "propagate": True
            },
            "httpx": {
                "level": "WARNING",
                "propagate": True
            }
        }
    }
    
    # Add format string for non-JSON formatter
    if not json_format:
        config["formatters"]["default"]["format"] = format_string
    
    logging.config.dictConfig(config)
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "level": level,
            "json_format": json_format,
            "log_file": log_file
        }
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)

# Context manager for adding request context to logs
class LogContext:
    """Context manager for adding context to log records."""
    
    def __init__(self, **context):
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory) 