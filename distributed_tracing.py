"""
Distributed Tracing for Observability
- Request correlation IDs
- Trace context propagation
- Latency breakdown by component
- Debug logging with correlation
"""

import logging
import threading
import time
import json
import uuid
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from collections import defaultdict
from contextvars import ContextVar

log = logging.getLogger("DISTRIBUTED-TRACING")

# Thread-safe context for trace ID
trace_context: ContextVar[Optional['TraceContext']] = ContextVar('trace_context', default=None)


@dataclass
class SpanMetrics:
    """Metrics for a single span."""
    start_time: float
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    status: str = "RUNNING"  # RUNNING, SUCCESS, ERROR


@dataclass
class Span:
    """Individual span in trace."""
    span_id: str
    parent_span_id: Optional[str]
    trace_id: str
    operation_name: str
    service_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    status: str = "RUNNING"
    error: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def finish(self, error: Optional[str] = None):
        """Mark span as finished."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = "ERROR" if error else "SUCCESS"
        if error:
            self.error = error

    def add_tag(self, key: str, value: Any):
        """Add tag to span."""
        self.tags[key] = value

    def add_log(self, message: str, **fields):
        """Add log entry to span."""
        self.logs.append({
            'timestamp': time.time(),
            'message': message,
            'fields': fields
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dict."""
        return {
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'trace_id': self.trace_id,
            'operation_name': self.operation_name,
            'service_name': self.service_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'status': self.status,
            'error': self.error,
            'tags': self.tags,
            'logs': self.logs
        }


@dataclass
class TraceContext:
    """Context for a trace."""
    trace_id: str
    service_name: str
    initial_span_id: str
    spans: Dict[str, Span] = field(default_factory=dict)
    root_span: Optional[Span] = None
    start_time: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def create_span(self, operation_name: str, parent_span_id: Optional[str] = None) -> Span:
        """Create new span in trace."""
        with self.lock:
            span_id = str(uuid.uuid4())

            span = Span(
                span_id=span_id,
                parent_span_id=parent_span_id or self.initial_span_id,
                trace_id=self.trace_id,
                operation_name=operation_name,
                service_name=self.service_name,
                start_time=time.time()
            )

            self.spans[span_id] = span
            return span

    def get_trace_tree(self) -> Dict[str, Any]:
        """Get trace as tree structure."""
        with self.lock:
            def build_tree(span_id):
                span = self.spans.get(span_id)
                if not span:
                    return None

                children = []
                for s in self.spans.values():
                    if s.parent_span_id == span_id:
                        child = build_tree(s.span_id)
                        if child:
                            children.append(child)

                return {
                    'span': span.to_dict(),
                    'children': children
                }

            return build_tree(self.initial_span_id) or {}

    def get_span_count(self) -> int:
        """Get total spans in trace."""
        with self.lock:
            return len(self.spans)

    def get_total_duration_ms(self) -> float:
        """Get total trace duration."""
        with self.lock:
            if not self.root_span:
                return 0.0
            return self.root_span.duration_ms


class Tracer:
    """
    Distributed tracer for request correlation and observability.
    
    Features:
    - Trace context propagation
    - Span creation and tracking
    - Trace storage and export
    """

    def __init__(self, service_name: str):
        """
        Initialize tracer.
        
        Args:
            service_name: Service name for spans
        """
        self.service_name = service_name
        self.traces: Dict[str, TraceContext] = {}
        self.active_traces = 0
        self.completed_traces = 0
        self.lock = threading.RLock()

    def start_trace(self, operation_name: str, parent_trace_id: Optional[str] = None) -> TraceContext:
        """
        Start new trace.
        
        Args:
            operation_name: Top-level operation
            parent_trace_id: Optional parent trace (for cross-service)
            
        Returns:
            TraceContext
        """
        with self.lock:
            trace_id = parent_trace_id or str(uuid.uuid4())
            initial_span_id = str(uuid.uuid4())

            root_span = Span(
                span_id=initial_span_id,
                parent_span_id=None,
                trace_id=trace_id,
                operation_name=operation_name,
                service_name=self.service_name,
                start_time=time.time()
            )

            context = TraceContext(
                trace_id=trace_id,
                service_name=self.service_name,
                initial_span_id=initial_span_id
            )

            context.spans[initial_span_id] = root_span
            context.root_span = root_span

            self.traces[trace_id] = context
            self.active_traces += 1

            # Set context variable for thread-local access
            token = trace_context.set(context)

            log.debug(f"Started trace {trace_id} for {operation_name}")

            return context

    def create_span(self, operation_name: str, 
                   parent_span_id: Optional[str] = None) -> Optional[Span]:
        """
        Create span in current trace.
        
        Args:
            operation_name: Operation name
            parent_span_id: Parent span ID (auto-detect current if None)
            
        Returns:
            New Span or None if no active trace
        """
        context = trace_context.get()

        if not context:
            return None

        return context.create_span(operation_name, parent_span_id)

    def finish_trace(self, trace_id: str, success: bool = True, error: Optional[str] = None):
        """
        Finish trace and store for export.
        
        Args:
            trace_id: Trace ID to finish
            success: Whether trace succeeded
            error: Optional error message
        """
        with self.lock:
            context = self.traces.get(trace_id)

            if not context:
                return

            if context.root_span:
                context.root_span.finish(error if not success else None)

            self.active_traces -= 1
            self.completed_traces += 1

            log.debug(f"Finished trace {trace_id} ({'SUCCESS' if success else 'ERROR'})")

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace by ID."""
        with self.lock:
            context = self.traces.get(trace_id)

            if not context:
                return None

            return {
                'trace_id': trace_id,
                'service': context.service_name,
                'span_count': context.get_span_count(),
                'duration_ms': context.get_total_duration_ms(),
                'tree': context.get_trace_tree()
            }

    def find_traces_by_operation(self, operation_name: str) -> List[str]:
        """Find traces by operation name."""
        with self.lock:
            matching = []

            for trace_id, context in self.traces.items():
                if context.root_span and context.root_span.operation_name == operation_name:
                    matching.append(trace_id)

            return matching

    def export_traces_json(self) -> str:
        """Export all traces as JSON."""
        with self.lock:
            traces_data = []

            for trace_id, context in self.traces.items():
                traces_data.append({
                    'trace_id': trace_id,
                    'service': context.service_name,
                    'spans': [s.to_dict() for s in context.spans.values()]
                })

            return json.dumps(traces_data, indent=2, default=str)

    def get_status(self) -> Dict[str, Any]:
        """Get tracer status."""
        with self.lock:
            return {
                'service': self.service_name,
                'active_traces': self.active_traces,
                'completed_traces': self.completed_traces,
                'trace_count': len(self.traces)
            }


class TracedLogger:
    """
    Logger that automatically includes trace context.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize traced logger.
        
        Args:
            logger: Base logger
        """
        self.logger = logger

    def _get_trace_prefix(self) -> str:
        """Get trace prefix for logging."""
        context = trace_context.get()

        if not context:
            return ""

        span_count = context.get_span_count()
        return f"[{context.trace_id[:8]}...{span_count} spans] "

    def debug(self, message: str, **kwargs):
        """Log debug with trace context."""
        self.logger.debug(self._get_trace_prefix() + message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info with trace context."""
        self.logger.info(self._get_trace_prefix() + message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning with trace context."""
        self.logger.warning(self._get_trace_prefix() + message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error with trace context."""
        self.logger.error(self._get_trace_prefix() + message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical with trace context."""
        self.logger.critical(self._get_trace_prefix() + message, **kwargs)
