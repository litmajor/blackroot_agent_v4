"""
Structured logging for Blackroot - ELK-ready centralized logging.

Features:
- JSON structured logging with context
- Log aggregation fields (agent_id, mission_id, cost, etc.)
- Performance metrics logging
- Error tracking with stack traces
- Log rotation and archival
"""

import logging
import json
import traceback
from typing import Any, Dict, Optional
from datetime import datetime
import sys
from logging.handlers import RotatingFileHandler
import os


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs compatible with ELK stacks.
    """
    
    def __init__(self, service_name: str = "blackroot"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service_name,
            "message": record.getMessage(),
            "source": {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName
            }
        }
        
        # Add exception info if present (defensive)
        if record.exc_info:
            exc_type = None
            exc_msg = None
            try:
                exc_type = record.exc_info[0] if record.exc_info and record.exc_info[0] else None
                exc_msg = record.exc_info[1] if record.exc_info and len(record.exc_info) > 1 else None
            except Exception:
                exc_type = None
                exc_msg = None

            log_data["exception"] = {
                "type": getattr(exc_type, "__name__", str(exc_type)),
                "message": str(exc_msg) if exc_msg is not None else "",
                "stack_trace": traceback.format_exc()
            }
        
        # Add custom attributes from log record
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName', 'relativeCreated',
                'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                'getMessage', 'asctime', 'taskName'
            ]:
                if not key.startswith('_'):
                    log_data[key] = self._serialize(value)
        
        return json.dumps(log_data, default=str)
    
    @staticmethod
    def _serialize(obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [StructuredFormatter._serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: StructuredFormatter._serialize(v) for k, v in obj.items()}
        else:
            return str(obj)


class ContextualLogger:
    """
    Wrapper around Python logger that adds structured context.
    
    Usage:
        logger = ContextualLogger("agent")
        logger.info("Mission started", agent_id="SCOUT-1", mission_id="m123", cost=5.0)
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """Set context fields that will be included in all logs."""
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context fields."""
        self.context.clear()
    
    def _add_context(self, extra: Optional[Dict] = None) -> Dict:
        """Merge context with extra fields."""
        merged = dict(self.context)
        if extra:
            merged.update(extra)
        return {"extra": merged} if merged else {}
    
    def debug(self, msg: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(msg, **self._add_context(kwargs))
    
    def info(self, msg: str, **kwargs):
        """Log info message with context."""
        self.logger.info(msg, **self._add_context(kwargs))
    
    def warning(self, msg: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(msg, **self._add_context(kwargs))
    
    def error(self, msg: str, exc_info: bool = False, **kwargs):
        """Log error message with context."""
        self.logger.error(msg, exc_info=exc_info, **self._add_context(kwargs))
    
    def critical(self, msg: str, exc_info: bool = False, **kwargs):
        """Log critical message with context."""
        self.logger.critical(msg, exc_info=exc_info, **self._add_context(kwargs))


class PerformanceLogger:
    """Logger for performance metrics and timings."""
    
    def __init__(self, name: str = "blackroot.performance"):
        self.logger = ContextualLogger(name)
    
    def log_operation(
        self,
        operation_name: str,
        duration_ms: float,
        success: bool = True,
        agent_id: Optional[str] = None,
        mission_id: Optional[str] = None,
        **kwargs
    ):
        """Log operation performance."""
        context = {
            "operation": operation_name,
            "duration_ms": round(duration_ms, 2),
            "status": "success" if success else "failure"
        }
        
        if agent_id:
            context["agent_id"] = agent_id
        if mission_id:
            context["mission_id"] = mission_id
        
        context.update(kwargs)
        
        level = "info" if success else "warning"
        msg = f"Operation {operation_name}: {duration_ms:.2f}ms"
        
        if level == "info":
            self.logger.info(msg, **context)
        else:
            self.logger.warning(msg, **context)


class ErrorLogger:
    """Logger for structured error tracking."""
    
    def __init__(self, name: str = "blackroot.errors"):
        self.logger = ContextualLogger(name)
    
    def log_error(
        self,
        error: Exception,
        context_msg: str,
        agent_id: Optional[str] = None,
        mission_id: Optional[str] = None,
        severity: str = "error",
        **kwargs
    ):
        """Log error with structured context."""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity
        }
        
        if agent_id:
            error_context["agent_id"] = agent_id
        if mission_id:
            error_context["mission_id"] = mission_id
        
        error_context.update(kwargs)
        
        if severity == "critical":
            self.logger.critical(context_msg, exc_info=True, **error_context)
        else:
            self.logger.error(context_msg, exc_info=True, **error_context)


def setup_structured_logging(
    service_name: str = "blackroot",
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    json_output: bool = True
) -> Dict[str, logging.Logger]:
    """
    Set up structured logging for the entire system.
    
    Returns:
        Dict with loggers for different components
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    structured_formatter = StructuredFormatter(service_name)
    simple_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler (human-readable or JSON based on config)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = structured_formatter if json_output else simple_formatter
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (JSON format)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "blackroot-structured.log"),
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(structured_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_file_handler = RotatingFileHandler(
        os.path.join(log_dir, "blackroot-errors.log"),
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(structured_formatter)
    root_logger.addHandler(error_file_handler)
    
    # Component-specific loggers
    loggers = {
        "root": root_logger,
        "contextual": ContextualLogger("blackroot"),
        "performance": PerformanceLogger(),
        "errors": ErrorLogger(),
        "agents": logging.getLogger("blackroot.agents"),
        "missions": logging.getLogger("blackroot.missions"),
        "resilience": logging.getLogger("blackroot.resilience"),
        "api": logging.getLogger("blackroot.api")
    }
    
    root_logger.info(
        f"Structured logging initialized (service={service_name}, json_output={json_output})"
    )
    
    return loggers


# Global logger instances
_loggers: Optional[Dict[str, Any]] = None


def get_logger(component: str = "root") -> logging.Logger:
    """Get a logger for a specific component."""
    global _loggers
    
    if _loggers is None:
        _loggers = setup_structured_logging()
    
    if component in _loggers:
        return _loggers[component]
    else:
        return logging.getLogger(f"blackroot.{component}")


def get_contextual_logger(name: str) -> ContextualLogger:
    """Get a contextual logger."""
    return ContextualLogger(f"blackroot.{name}")


def log_mission_event(
    mission_id: str,
    event_type: str,
    agent_id: str,
    status: str,
    cost: float = 0.0,
    **kwargs
):
    """Log a mission-related event."""
    logger = get_contextual_logger("missions")
    logger.set_context(
        mission_id=mission_id,
        agent_id=agent_id,
        event_type=event_type,
        cost=cost
    )
    logger.info(f"Mission event: {event_type} - {status}", status=status, **kwargs)
    logger.clear_context()
