"""
Reliable Message Delivery for BlackRoot SwarmMesh
- Implements Request Waiting List pattern for delivery tracking
- Deduplicates messages based on message ID + sequence number
- Guarantees at-least-once delivery with request waiting
- Provides retry logic with exponential backoff
"""

import time
import threading
import logging
from typing import Dict, Optional, Any, Callable, Deque
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import uuid

log = logging.getLogger("RELIABLE-DELIVERY")


class DeliveryStatus(Enum):
    """Status of a message delivery."""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    DUPLICATE = "duplicate"


@dataclass
class Message:
    """Message wrapper with delivery tracking."""
    message_id: str
    sequence_number: int
    destination: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    delivery_attempts: int = 0
    status: DeliveryStatus = DeliveryStatus.PENDING
    last_attempt_time: Optional[float] = None
    acknowledged_at: Optional[float] = None

    def to_dict(self):
        return {
            'message_id': self.message_id,
            'sequence_number': self.sequence_number,
            'destination': self.destination,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'delivery_attempts': self.delivery_attempts,
            'status': self.status.value
        }


@dataclass
class RequestWaitingListEntry:
    """Entry in request waiting list (tracks pending requests)."""
    request_id: str
    sequence_number: int
    sender: str
    receiver: str
    timeout_ms: float
    created_at: float = field(default_factory=time.time)
    acknowledged_at: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if request has timed out."""
        elapsed_ms = (time.time() - self.created_at) * 1000
        return elapsed_ms > self.timeout_ms

    def is_acknowledged(self) -> bool:
        """Check if request has been acknowledged."""
        return self.acknowledged_at is not None


class DuplicateDetector:
    """
    Detects duplicate messages based on message ID and sequence number.
    Maintains sliding window of seen messages.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize duplicate detector.
        
        Args:
            window_size: Maximum number of message IDs to track
        """
        self.window_size = window_size
        self.seen_messages: Dict[str, int] = {}  # message_id -> sequence_number
        self.lock = threading.RLock()

    def is_duplicate(self, message_id: str, sequence_number: int) -> bool:
        """Check if message is a duplicate."""
        with self.lock:
            if message_id in self.seen_messages:
                return self.seen_messages[message_id] >= sequence_number
            return False

    def record_message(self, message_id: str, sequence_number: int):
        """Record message as seen."""
        with self.lock:
            self.seen_messages[message_id] = sequence_number

            # Trim window if too large
            if len(self.seen_messages) > self.window_size:
                # Remove oldest entries
                oldest_keys = sorted(
                    self.seen_messages.keys(),
                    key=lambda k: self.seen_messages[k]
                )[:len(self.seen_messages) - self.window_size]
                for key in oldest_keys:
                    del self.seen_messages[key]

    def forget_message(self, message_id: str):
        """Remove message from tracking."""
        with self.lock:
            self.seen_messages.pop(message_id, None)


class RequestWaitingList:
    """
    Tracks pending requests to guarantee delivery.
    
    Pattern: Each message gets a unique request ID + sequence number.
    Receiver must acknowledge with same ID/sequence to confirm delivery.
    """

    def __init__(self, timeout_ms: float = 5000):
        """
        Initialize request waiting list.
        
        Args:
            timeout_ms: Default timeout for requests
        """
        self.timeout_ms = timeout_ms
        self.requests: Dict[str, RequestWaitingListEntry] = {}
        self.lock = threading.RLock()

    def add_request(self, sender: str, receiver: str, sequence_number: int) -> str:
        """
        Add new request to waiting list.
        
        Returns:
            request_id (unique ID for this request)
        """
        request_id = f"{sender}:{receiver}:{sequence_number}:{int(time.time() * 1000)}"

        entry = RequestWaitingListEntry(
            request_id=request_id,
            sequence_number=sequence_number,
            sender=sender,
            receiver=receiver,
            timeout_ms=self.timeout_ms
        )

        with self.lock:
            self.requests[request_id] = entry

        log.debug(f"Added request to waiting list: {request_id}")
        return request_id

    def acknowledge_request(self, request_id: str) -> bool:
        """
        Mark request as acknowledged.
        
        Returns:
            True if request was in waiting list and acknowledged
        """
        with self.lock:
            if request_id not in self.requests:
                log.warning(f"Acknowledgement for unknown request: {request_id}")
                return False

            entry = self.requests[request_id]
            entry.acknowledged_at = time.time()
            log.debug(f"Acknowledged request: {request_id}")
            return True

    def get_pending_requests(self) -> list[RequestWaitingListEntry]:
        """Get all pending (unacknowledged) requests."""
        with self.lock:
            return [
                entry for entry in self.requests.values()
                if not entry.is_acknowledged() and not entry.is_expired()
            ]

    def get_expired_requests(self) -> list[RequestWaitingListEntry]:
        """Get all expired requests (need retry or failure handling)."""
        with self.lock:
            return [
                entry for entry in self.requests.values()
                if not entry.is_acknowledged() and entry.is_expired()
            ]

    def cleanup_expired(self) -> int:
        """Remove expired requests from tracking."""
        with self.lock:
            expired = [
                request_id for request_id, entry in self.requests.items()
                if entry.is_acknowledged() or (entry.is_expired() and 
                    (time.time() - entry.created_at) > self.timeout_ms * 2)
            ]

            for request_id in expired:
                del self.requests[request_id]

            return len(expired)

    def get_status(self) -> Dict[str, Any]:
        """Get status of waiting list."""
        with self.lock:
            pending = sum(1 for e in self.requests.values() if not e.is_acknowledged())
            expired = sum(1 for e in self.requests.values() if e.is_expired())

            return {
                'total_requests': len(self.requests),
                'pending': pending,
                'acknowledged': len(self.requests) - pending,
                'expired': expired
            }


class ReliableDeliveryManager:
    """
    Manages reliable message delivery with deduplication and retry logic.
    """

    def __init__(self, max_retries: int = 5, base_backoff_ms: float = 100):
        """
        Initialize delivery manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_backoff_ms: Base backoff time for exponential backoff
        """
        self.max_retries = max_retries
        self.base_backoff_ms = base_backoff_ms
        self.duplicate_detector = DuplicateDetector()
        self.request_waiting_list = RequestWaitingList()
        self.pending_messages: Deque[Message] = deque(maxlen=10000)
        self.delivered_messages: Deque[Message] = deque(maxlen=1000)
        self.lock = threading.RLock()
        self.sequence_counter: Dict[str, int] = {}  # sender -> sequence

    def queue_message(self, destination: str, payload: Any, sender: str = "system") -> str:
        """
        Queue message for delivery.
        
        Returns:
            message_id
        """
        with self.lock:
            if sender not in self.sequence_counter:
                self.sequence_counter[sender] = 0
            self.sequence_counter[sender] += 1

            message_id = str(uuid.uuid4())
            sequence_number = self.sequence_counter[sender]

            message = Message(
                message_id=message_id,
                sequence_number=sequence_number,
                destination=destination,
                payload=payload
            )

            self.pending_messages.append(message)
            self.request_waiting_list.add_request(sender, destination, sequence_number)

            log.info(f"Queued message {message_id} for {destination} (seq={sequence_number})")
            return message_id

    def deliver_message(self, message: Message, delivery_fn: Callable) -> bool:
        """
        Attempt to deliver message.
        
        Args:
            message: Message to deliver
            delivery_fn: Callable to perform actual delivery
            
        Returns:
            True if delivery succeeded
        """
        with self.lock:
            # Check for duplicate
            if self.duplicate_detector.is_duplicate(message.message_id, message.sequence_number):
                message.status = DeliveryStatus.DUPLICATE
                log.warning(f"Duplicate message detected: {message.message_id}")
                return False

            message.delivery_attempts += 1
            message.last_attempt_time = time.time()

            try:
                # Perform delivery
                success = delivery_fn(message)

                if success:
                    message.status = DeliveryStatus.ACKNOWLEDGED
                    message.acknowledged_at = time.time()
                    self.duplicate_detector.record_message(message.message_id, message.sequence_number)
                    self.delivered_messages.append(message)
                    self.request_waiting_list.acknowledge_request(message.message_id)
                    log.info(f"Delivered message {message.message_id} (attempt {message.delivery_attempts})")
                    return True
                else:
                    message.status = DeliveryStatus.SENT
                    log.debug(f"Delivery attempt {message.delivery_attempts} for {message.message_id}")
                    return False

            except Exception as e:
                message.status = DeliveryStatus.SENT
                log.error(f"Delivery error for {message.message_id}: {e}")
                return False

    def get_retry_backoff_ms(self, delivery_attempts: int) -> float:
        """Calculate exponential backoff for retry."""
        if delivery_attempts > self.max_retries:
            return -1  # No more retries

        # Exponential backoff: backoff = base * 2^(attempts-1)
        backoff = self.base_backoff_ms * (2 ** (delivery_attempts - 1))
        # Cap at 30 seconds
        return min(backoff, 30000)

    def should_retry(self, message: Message) -> bool:
        """Check if message should be retried."""
        elapsed_ms = (time.time() - (message.last_attempt_time or message.timestamp)) * 1000
        backoff_ms = self.get_retry_backoff_ms(message.delivery_attempts)

        if backoff_ms < 0:
            return False

        return elapsed_ms >= backoff_ms

    def get_status(self) -> Dict[str, Any]:
        """Get delivery manager status."""
        with self.lock:
            pending = [m for m in self.pending_messages if m.status == DeliveryStatus.PENDING]
            failed = [m for m in self.pending_messages if m.status == DeliveryStatus.FAILED]

            return {
                'pending_messages': len(pending),
                'failed_messages': len(failed),
                'delivered_messages': len(self.delivered_messages),
                'duplicate_window_size': len(self.duplicate_detector.seen_messages),
                'request_waiting_list': self.request_waiting_list.get_status()
            }
