#!/usr/bin/env python3
"""
Propagation Retry Scheduler with Exponential Backoff
Handles failed propagation attempts gracefully with DLQ support
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("RetryScheduler")


class PropagationStatus(Enum):
    """Propagation attempt status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY_SCHEDULED = "retry_scheduled"
    DEAD_LETTER = "dead_letter"  # Permanently failed


@dataclass
class PropagationAttempt:
    """Single propagation attempt record"""
    target_id: str  # hostname or IP
    credential_id: str  # domain\username
    attempt_number: int
    status: PropagationStatus
    timestamp: str
    next_retry: Optional[str]
    error_message: Optional[str]
    retry_count: int = 0
    max_retries: int = 5
    base_backoff: float = 2.0  # seconds
    jitter_factor: float = 0.1  # ±10% randomness


class ExponentialBackoffScheduler:
    """Schedule retries with exponential backoff and jitter"""
    
    def __init__(self, base_backoff: float = 2.0, max_backoff: float = 3600.0, jitter: bool = True):
        """
        Args:
            base_backoff: Initial backoff time in seconds (default: 2s)
            max_backoff: Maximum backoff time in seconds (default: 1 hour)
            jitter: Add randomness to prevent thundering herd
        """
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter
    
    def calculate_backoff(self, attempt_number: int) -> float:
        """Calculate backoff time for attempt number
        
        Formula: base * (2 ^ attempt) with optional jitter
        Example:
            Attempt 1: 2s
            Attempt 2: 4s
            Attempt 3: 8s
            Attempt 4: 16s
            Attempt 5: 32s (capped at max_backoff)
        """
        backoff = self.base_backoff * (2 ** attempt_number)
        backoff = min(backoff, self.max_backoff)
        
        if self.jitter:
            import random
            jitter_amount = backoff * 0.1  # ±10%
            backoff += random.uniform(-jitter_amount, jitter_amount)
        
        return max(backoff, self.base_backoff)  # Never go below base
    
    def schedule_next_retry(self, attempt: PropagationAttempt) -> Optional[datetime]:
        """Calculate next retry time"""
        if attempt.retry_count >= attempt.max_retries:
            return None  # No more retries
        
        backoff = self.calculate_backoff(attempt.retry_count)
        next_retry = datetime.utcnow() + timedelta(seconds=backoff)
        
        return next_retry


class PropagationRetryQueue:
    """Queue and track propagation retry attempts"""
    
    def __init__(self, queue_file: str = "propagation_queue.json", dlq_file: str = "propagation_dlq.json"):
        self.queue_file = Path(queue_file)
        self.dlq_file = Path(dlq_file)
        self.queue: List[PropagationAttempt] = []
        self.dlq: List[PropagationAttempt] = []
        self.scheduler = ExponentialBackoffScheduler()
        
        self._load_queues()
    
    def _load_queues(self):
        """Load queue from disk"""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'r') as f:
                    data = json.load(f)
                    self.queue = [self._dict_to_attempt(d) for d in data.get("queue", [])]
                    log.info(f"✅ Loaded {len(self.queue)} pending retries from {self.queue_file}")
            except Exception as e:
                log.warning(f"⚠️  Failed to load queue: {e}")
        
        if self.dlq_file.exists():
            try:
                with open(self.dlq_file, 'r') as f:
                    data = json.load(f)
                    self.dlq = [self._dict_to_attempt(d) for d in data.get("dlq", [])]
                    log.info(f"⚠️  {len(self.dlq)} permanently failed propagations in DLQ")
            except Exception as e:
                log.warning(f"⚠️  Failed to load DLQ: {e}")
    
    def _attempt_to_dict(self, attempt: PropagationAttempt) -> Dict:
        """Convert PropagationAttempt to JSON-serializable dict"""
        d = asdict(attempt)
        # Convert enum to string value
        d['status'] = attempt.status.value
        return d
    
    def _save_queues(self):
        """Save queues to disk"""
        with open(self.queue_file, 'w') as f:
            json.dump(
                {"timestamp": datetime.utcnow().isoformat(), "queue": [self._attempt_to_dict(a) for a in self.queue]},
                f, indent=2
            )
        
        with open(self.dlq_file, 'w') as f:
            json.dump(
                {"timestamp": datetime.utcnow().isoformat(), "dlq": [self._attempt_to_dict(a) for a in self.dlq]},
                f, indent=2
            )
    
    def _dict_to_attempt(self, d: Dict) -> PropagationAttempt:
        """Convert dict to PropagationAttempt"""
        d['status'] = PropagationStatus(d.get('status', 'pending'))
        return PropagationAttempt(**d)
    
    def add_attempt(self, target_id: str, credential_id: str, error_message: Optional[str] = None):
        """Add failed propagation attempt to queue"""
        attempt = PropagationAttempt(
            target_id=target_id,
            credential_id=credential_id,
            attempt_number=0,
            status=PropagationStatus.RETRY_SCHEDULED,
            timestamp=datetime.utcnow().isoformat(),
            next_retry=None,
            error_message=error_message,
            retry_count=0
        )
        
        # Schedule next retry
        next_retry = self.scheduler.schedule_next_retry(attempt)
        if next_retry:
            attempt.next_retry = next_retry.isoformat()
            retry_seconds = (next_retry - datetime.utcnow()).total_seconds()
            log.info(f"📋 Queued retry: {target_id} via {credential_id} (retry in {retry_seconds}s)")
            self.queue.append(attempt)
        else:
            log.warning(f"❌ Max retries exceeded for {target_id}, adding to DLQ")
            attempt.status = PropagationStatus.DEAD_LETTER
            self.dlq.append(attempt)
        
        self._save_queues()
    
    def get_ready_for_retry(self) -> List[PropagationAttempt]:
        """Get attempts ready for retry (overdue)"""
        now = datetime.utcnow()
        ready = []
        
        for attempt in self.queue:
            if attempt.next_retry:
                next_retry_time = datetime.fromisoformat(attempt.next_retry.replace('Z', '+00:00'))
                if next_retry_time <= now:
                    ready.append(attempt)
        
        return ready
    
    def mark_success(self, target_id: str, credential_id: str):
        """Mark propagation as successful and remove from queue"""
        self.queue = [a for a in self.queue if not (a.target_id == target_id and a.credential_id == credential_id)]
        self._save_queues()
        log.info(f"✅ Propagation success: {target_id}")
    
    def mark_failed(self, target_id: str, credential_id: str, error_message: str):
        """Mark propagation as failed, reschedule or move to DLQ"""
        # Find attempt
        attempt_idx = None
        for i, a in enumerate(self.queue):
            if a.target_id == target_id and a.credential_id == credential_id:
                attempt_idx = i
                break
        
        if attempt_idx is None:
            log.warning(f"⚠️  Attempt not found: {target_id}/{credential_id}")
            return
        
        attempt = self.queue[attempt_idx]
        attempt.retry_count += 1
        attempt.error_message = error_message
        attempt.timestamp = datetime.utcnow().isoformat()
        
        if attempt.retry_count >= attempt.max_retries:
            # Move to DLQ (permanent failure)
            attempt.status = PropagationStatus.DEAD_LETTER
            self.dlq.append(attempt)
            self.queue.pop(attempt_idx)
            log.warning(f"💀 Permanently failed (DLQ): {target_id} via {credential_id}")
        else:
            # Reschedule
            next_retry = self.scheduler.schedule_next_retry(attempt)
            attempt.next_retry = next_retry.isoformat() if next_retry else None
            attempt.status = PropagationStatus.RETRY_SCHEDULED
            log.info(f"🔄 Rescheduled: {target_id} (attempt {attempt.retry_count}/{attempt.max_retries})")
        
        self._save_queues()
    
    def get_stats(self) -> Dict:
        """Get queue statistics"""
        now = datetime.utcnow()
        
        pending = len(self.queue)
        ready = len(self.get_ready_for_retry())
        forever_failed = len(self.dlq)
        
        # Calculate next retry times
        next_retries = []
        for attempt in self.queue:
            if attempt.next_retry:
                next_retry_time = datetime.fromisoformat(attempt.next_retry.replace('Z', '+00:00'))
                next_retries.append((next_retry_time - now).total_seconds())
        
        avg_wait = sum(next_retries) / len(next_retries) if next_retries else 0
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "pending_retries": pending,
            "ready_for_retry": ready,
            "dead_letter_queue": forever_failed,
            "average_wait_seconds": avg_wait,
            "queue_file": str(self.queue_file),
            "dlq_file": str(self.dlq_file)
        }


class PropagationRetryController:
    """Main controller for retry logic"""
    
    def __init__(self, propagation_callback: Callable):
        """
        Args:
            propagation_callback: Function to call for retry attempts
                                  Signature: callback(target_id, credential_id) -> bool
        """
        self.queue = PropagationRetryQueue()
        self.callback = propagation_callback
        self.running = False
    
    def handle_failed_propagation(self, target_id: str, credential_id: str, error: str):
        """Handle failed propagation attempt"""
        self.queue.add_attempt(target_id, credential_id, error)
    
    def retry_loop(self, interval_seconds: int = 60):
        """Main retry loop (run in background thread)
        
        Args:
            interval_seconds: How often to check for ready retries
        """
        self.running = True
        
        log.info(f"🔄 Retry loop started (check interval: {interval_seconds}s)")
        
        while self.running:
            try:
                # Get attempts ready for retry
                ready = self.queue.get_ready_for_retry()
                
                if ready:
                    log.info(f"📡 {len(ready)} propagation(s) ready for retry")
                    
                    for attempt in ready:
                        log.info(f"🔄 Retrying: {attempt.target_id} via {attempt.credential_id}")
                        
                        try:
                            success = self.callback(attempt.target_id, attempt.credential_id)
                            
                            if success:
                                self.queue.mark_success(attempt.target_id, attempt.credential_id)
                            else:
                                self.queue.mark_failed(attempt.target_id, attempt.credential_id, "Retry failed")
                        
                        except Exception as e:
                            self.queue.mark_failed(attempt.target_id, attempt.credential_id, str(e))
                
                # Print stats
                stats = self.queue.get_stats()
                if stats["pending_retries"] > 0:
                    log.info(f"📊 Retry Queue: {stats['pending_retries']} pending, "
                           f"{stats['ready_for_retry']} ready, {stats['dead_letter_queue']} DLQ")
                
                time.sleep(interval_seconds)
            
            except Exception as e:
                log.error(f"❌ Retry loop error: {e}")
                time.sleep(interval_seconds)
    
    def stop(self):
        """Stop retry loop"""
        self.running = False
        log.info("🛑 Retry loop stopped")
    
    def get_dlq_entries(self) -> List[Dict]:
        """Get permanently failed propagations"""
        return [asdict(a) for a in self.queue.dlq]


if __name__ == "__main__":
    # Demo
    def demo_retry_callback(target_id: str, cred_id: str) -> bool:
        import random
        success = random.random() > 0.7  # 30% success rate
        log.info(f"  → Retry attempt: {success}")
        return success
    
    controller = PropagationRetryController(demo_retry_callback)
    
    # Simulate some failed propagations
    controller.handle_failed_propagation("server1.internal", "DOMAIN\\admin", "Connection refused")
    controller.handle_failed_propagation("server2.internal", "DOMAIN\\user", "Authentication failed")
    
    # Show stats
    stats = controller.queue.get_stats()
    print(f"\n📊 Queue Stats:")
    print(f"  Pending retries: {stats['pending_retries']}")
    print(f"  Ready for retry: {stats['ready_for_retry']}")
    print(f"  DLQ entries: {stats['dead_letter_queue']}")
