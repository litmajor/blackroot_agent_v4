"""
ADAPTIVE PERIODIC TASK RUNNER - Daemon Management Layer

Provides thread-safe, adaptive periodic task execution with:
- Configurable base intervals
- Exponential backoff on errors
- Hot-swappable task functions
- Graceful shutdown with timeout
- Optional jitter (Phase 3 enhancement)

Used by: EnforcerAgent, SwarmMesh (future), Kernel (future), PeerLinker (future)
"""

import threading
import logging
from typing import Callable, Optional

log = logging.getLogger("TASK-RUNNER")


class TaskRunner:
    """
    Adaptive periodic task runner with exponential backoff.

    Task function may:
        - return None -> use default interval
        - return float -> override next interval (seconds)
        - raise Exception -> triggers exponential backoff

    Thread-safe. Daemon threads by default.
    """

    def __init__(
        self,
        task_fn: Callable[[], Optional[float]],
        interval: float = 1.0,
        backoff_factor: float = 2.0,
        max_backoff: float = 60.0,
        name: str = "TaskRunner",
    ):
        """
        Initialize TaskRunner.

        Args:
            task_fn: Callable that returns Optional[float] (next interval).
            interval: Default interval in seconds.
            backoff_factor: Multiplier for exponential backoff (e.g., 2.0 = double on each error).
            max_backoff: Maximum backoff cap in seconds.
            name: Name used in logging and thread identification.
        """
        self._task_fn = task_fn
        self._default_interval = interval
        self._current_interval = interval
        self._backoff_factor = backoff_factor
        self._max_backoff = max_backoff
        self._name = name

        self._running = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._execution_count = 0
        self._error_count = 0

    # =====================
    # Public Lifecycle API
    # =====================

    def start(self, run_immediately: bool = False) -> None:
        """
        Start the task runner daemon.

        Args:
            run_immediately: If True, spawn a thread to run task once before main loop starts.
        """
        with self._lock:
            if self._running:
                log.debug(f"[{self._name}] Already running; ignoring start() call")
                return
            self._running = True
            self._stop_event.clear()
            self._current_interval = self._default_interval
            self._execution_count = 0
            self._error_count = 0

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=self._name)
        self._thread.start()
        log.debug(f"[{self._name}] Started (interval={self._default_interval}s, backoff={self._backoff_factor})")

        if run_immediately:
            threading.Thread(
                target=self._safe_execute,
                daemon=True,
                name=f"{self._name}-immediate",
            ).start()

    def stop(self, timeout: Optional[float] = None) -> bool:
        """
        Stop the task runner gracefully.

        Args:
            timeout: Maximum seconds to wait for thread to exit. None = no timeout.

        Returns:
            True if stopped cleanly, False if timeout exceeded.
        """
        with self._lock:
            was_running = self._running
            self._running = False
            self._stop_event.set()

        if not was_running or not self._thread:
            return True

        self._thread.join(timeout=timeout)
        is_clean = not self._thread.is_alive()
        
        if is_clean:
            log.debug(f"[{self._name}] Stopped cleanly (executed {self._execution_count} times, {self._error_count} errors)")
        else:
            log.warning(f"[{self._name}] Timeout during shutdown; thread still alive")
        
        return is_clean

    def is_running(self) -> bool:
        """Check if runner is currently active."""
        with self._lock:
            return self._running

    # =====================
    # Task Control API
    # =====================

    def update_task(self, new_task_fn: Callable[[], Optional[float]]) -> None:
        """
        Hot-swap the task function (thread-safe).

        Useful for updating behavior without restarting the runner.
        """
        with self._lock:
            self._task_fn = new_task_fn
            log.debug(f"[{self._name}] Task function updated")

    def reset_backoff(self) -> None:
        """Reset interval to default (useful after one-off recovery)."""
        with self._lock:
            self._current_interval = self._default_interval

    def get_stats(self) -> dict:
        """Return execution statistics."""
        with self._lock:
            return {
                "name": self._name,
                "running": self._running,
                "execution_count": self._execution_count,
                "error_count": self._error_count,
                "current_interval": self._current_interval,
                "default_interval": self._default_interval,
            }

    # =====================
    # Internal Execution
    # =====================

    def _run_loop(self) -> None:
        """Main event loop (runs in daemon thread)."""
        log.debug(f"[{self._name}] Monitor loop started")
        try:
            while not self._stop_event.wait(timeout=self._current_interval):
                self._safe_execute()
        finally:
            log.debug(f"[{self._name}] Monitor loop exited")

    def _safe_execute(self) -> None:
        """Execute task with error handling and backoff."""
        try:
            with self._lock:
                task = self._task_fn
                self._execution_count += 1

            result = task()

            # On success, reset to default interval (or use returned interval)
            with self._lock:
                if isinstance(result, (int, float)):
                    self._current_interval = result
                else:
                    self._current_interval = self._default_interval

        except Exception as e:
            log.error(f"[{self._name}] Task error: {e}")

            # Exponential backoff on failure
            with self._lock:
                self._error_count += 1
                self._current_interval = min(
                    self._current_interval * self._backoff_factor,
                    self._max_backoff,
                )
                log.debug(f"[{self._name}] Backoff: next interval = {self._current_interval}s")


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level convenience functions
# ═══════════════════════════════════════════════════════════════════════════════

def create_runner(
    task_fn: Callable[[], Optional[float]],
    interval: float = 1.0,
    name: str = "TaskRunner",
    **kwargs
) -> TaskRunner:
    """
    Factory function for creating and starting a TaskRunner.

    Example:
        runner = create_runner(my_task_fn, interval=5.0, name="MyDaemon")
        # runner is already started
    """
    runner = TaskRunner(task_fn, interval=interval, name=name, **kwargs)
    runner.start()
    return runner


def get_task_runner_template() -> str:
    """Return a template for converting async monitors to TaskRunner."""
    return """
# Template: Convert async while-True monitor to TaskRunner

# BEFORE (async):
async def monitor_something(self):
    while True:
        try:
            result = await fetch_data()
            process(result)
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(e)
            await asyncio.sleep(5)  # backoff

# AFTER (TaskRunner):
def _monitor_task(self) -> Optional[float]:
    '''Non-blocking task function for TaskRunner.'''
    try:
        result = fetch_data()  # Synchronous call
        process(result)
        return None  # Use default interval (1.0s)
    except Exception as e:
        logger.error(e)
        return 5.0  # Override interval for backoff

# In __init__:
self._monitor_runner = TaskRunner(
    self._monitor_task,
    interval=1.0,
    name="compliance-monitor"
)
self._monitor_runner.start()

# In shutdown:
self._monitor_runner.stop(timeout=5.0)
"""
