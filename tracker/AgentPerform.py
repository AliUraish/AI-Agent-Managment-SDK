import time
import uuid
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Deque
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, OrderedDict
import requests
import aiohttp
from urllib.parse import urljoin
import re

# Import secure components from AgentOper
from .AgentOper import SecureLogger, SecureAPIClient, APIResponse


@dataclass
class QueuedEvent:
    """Represents an event queued for later backend replay"""
    event_type: str  # "start", "end", "failed", "expired"
    timestamp: datetime
    data: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


class ConversationQuality(Enum):
    """Enumeration for conversation quality ratings"""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    VERY_POOR = 1

    @classmethod
    def from_int(cls, value: int) -> 'ConversationQuality':
        """Convert integer to ConversationQuality enum"""
        for quality in cls:
            if quality.value == value:
                return quality
        raise ValueError(f"Invalid quality score: {value}. Must be 1-5.")

    @classmethod
    def from_int_safe(cls, value: int) -> Optional['ConversationQuality']:
        """Safely convert integer to ConversationQuality enum, returns None if invalid"""
        try:
            return cls.from_int(value)
        except ValueError:
            return None


@dataclass
class SessionInfo:
    """Lightweight session information for quick validation (hybrid model)"""
    agent_id: str
    start_time: datetime
    run_id: str  # Unique identifier for this conversation run
    user_id: Optional[str] = None
    last_access_time: Optional[datetime] = None  # For sliding TTL
    is_ended: bool = False  # Track if conversation has ended locally
    
    def __post_init__(self):
        """Initialize last_access_time if not provided"""
        if self.last_access_time is None:
            self.last_access_time = self.start_time
    
    def is_expired(self, ttl_hours: float) -> bool:
        """Check if session has expired based on sliding TTL (last access time)"""
        expiry_time = self.last_access_time + timedelta(hours=ttl_hours)
        return datetime.now() > expiry_time
    
    def touch(self):
        """Update last access time (sliding TTL)"""
        self.last_access_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        return {
            'agent_id': self.agent_id,
            'start_time': self.start_time.isoformat(),
            'run_id': self.run_id,
            'user_id': self.user_id,
            'last_access_time': self.last_access_time.isoformat() if self.last_access_time else None,
            'is_ended': self.is_ended
        }


@dataclass
class ConversationStartData:
    """Data for starting a conversation with hybrid tracking"""
    agent_id: str
    session_id: str
    run_id: str
    user_id: Optional[str] = None
    start_time: Optional[str] = None
    context: Optional[Dict[str, Any]] = None  # For session resumption
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ConversationEndData:
    """Data for ending a conversation with hybrid tracking"""
    agent_id: str
    session_id: str
    run_id: str
    end_time: Optional[str] = None
    duration_seconds: Optional[int] = None
    quality_score: Optional[Union[int, str]] = None
    user_feedback: Optional[str] = None
    message_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class FailedSessionData:
    """Data for recording failed sessions with hybrid tracking"""
    agent_id: str
    session_id: str
    run_id: str
    error_message: str
    failure_time: Optional[str] = None
    duration_seconds: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SessionRetrievalQuery:
    """Query parameters for session retrieval from backend"""
    session_id: str
    include_context: bool = True
    include_history: bool = False

@dataclass
class MessageData:
    """Data for individual message tracking"""
    session_id: str
    agent_id: str
    message_id: str
    timestamp: str
    message_type: str  # "user", "agent", "system"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    
@dataclass
class ConversationHistoryData:
    """Data for conversation history tracking"""
    session_id: str
    agent_id: str
    messages: List[Dict[str, Any]]
    conversation_summary: Optional[str] = None
    total_messages: int = 0
    total_tokens: int = 0
    start_time: str = ""
    last_update: str = ""
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking for agent operations"""
    total_sessions: int = 0
    successful_sessions: int = 0
    failed_sessions: int = 0
    total_response_time_ms: float = 0.0
    quality_scores: List[int] = None
    agent_failure_counts: Dict[str, int] = None
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = []
        if self.agent_failure_counts is None:
            self.agent_failure_counts = {}
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_sessions == 0:
            return 0.0
        return (self.successful_sessions / self.total_sessions) * 100.0
    
    @property
    def average_response_time_ms(self) -> float:
        """Calculate average response time"""
        if self.successful_sessions == 0:
            return 0.0
        return self.total_response_time_ms / self.successful_sessions
    
    @property
    def average_quality_score(self) -> float:
        """Calculate average quality score"""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)


class AgentPerformanceTracker:
    """
    Agent Performance Tracker - Handles performance metrics:
    - Success rates
    - Response time
    - Conversation quality
    - Failed sessions
    
    Now includes lightweight session tracking with TTL-based cleanup.
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 client_id: Optional[str] = None,
                 session_ttl_hours: float = 10.0,
                 cleanup_interval_minutes: int = 30,
                 backend_ttl_hours: float = 20.0,  # Backend persistence TTL
                 max_cache_size: int = 50000,  # Maximum cache size for LRU (increased for scalability)
                 max_offline_queue_size: int = 5000,  # Maximum offline queue size
                 batch_notification_size: int = 50,  # Batch size for expiry notifications
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Agent Performance Tracker with hybrid session management
        
        Args:
            base_url: API base URL
            api_key: API authentication key
            client_id: Client identifier for SDK instance
            session_ttl_hours: Local cache TTL (default: 10 hours, sliding)
            cleanup_interval_minutes: Cache cleanup interval (default: 30 minutes)
            backend_ttl_hours: Backend persistence TTL (default: 20 hours)
            max_cache_size: Maximum number of sessions in LRU cache (default: 50,000)
            max_offline_queue_size: Maximum number of events in offline queue
            batch_notification_size: Number of notifications to batch together
            logger: Optional logger instance
        """
        self.base_url = base_url.rstrip('/')
        self.client_id = client_id or f"sdk_client_{int(datetime.now().timestamp())}"
        self.session_ttl_hours = session_ttl_hours
        self.backend_ttl_hours = backend_ttl_hours
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.max_cache_size = max_cache_size
        self.max_offline_queue_size = max_offline_queue_size
        self.batch_notification_size = batch_notification_size
        
        # Setup secure logging
        self.logger = logger or logging.getLogger(__name__)
        self.secure_logger = SecureLogger()
        
        # Initialize API client
        self.api_client = SecureAPIClient(base_url, api_key)
        
        # Performance metrics tracking
        self._performance_metrics = PerformanceMetrics()
        self._metrics_lock = threading.RLock()
        
        # LRU Session cache for lightweight session tracking
        self._session_cache: OrderedDict[str, SessionInfo] = OrderedDict()
        self._cache_lock = threading.RLock()
        
        # Offline event queue for when backend is unreachable
        self._offline_queue: Deque[QueuedEvent] = deque()
        self._queue_lock = threading.RLock()
        
        # Batched notifications for expiry events
        self._pending_expiry_notifications: List[Dict[str, Any]] = []
        self._notification_lock = threading.RLock()
        
        # Backend status tracking
        self._backend_available = True
        self._last_backend_check = datetime.now()
        
        # Cleanup management
        self._cleanup_stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._async_cleanup_task: Optional[asyncio.Task] = None
        
        # Start TTL cleanup daemon thread
        self._start_cleanup_daemon()
        
        # Log initialization without exposing API key
        self.logger.info(
            self.secure_logger.format_log(
                "AgentPerformanceTracker initialized with base URL: %s, Hybrid Model: Local TTL=%.1fh, Backend TTL=%.1fh, Max Cache=%dk",
                self.base_url, self.session_ttl_hours, self.backend_ttl_hours, self.max_cache_size // 1000
            )
        )
    
    def _generate_session_id(self, agent_id: str, start_time: datetime) -> str:
        """Generate session ID in format: {agent_id}_{start_time_timestamp}_{random}"""
        timestamp = int(start_time.timestamp())
        random_suffix = str(uuid.uuid4())[:8]  # Short random identifier
        return f"{agent_id}_{timestamp}_{random_suffix}"
    
    def _start_cleanup_daemon(self):
        """Start the cleanup daemon thread"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_stop_event.clear()
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_daemon,
                daemon=True,
                name="SessionCleanupDaemon"
            )
            self._cleanup_thread.start()
            self.logger.info("Session cleanup daemon started")
    
    def _start_async_cleanup(self):
        """Start async cleanup task for async SDK usage"""
        if self._async_cleanup_task is None or self._async_cleanup_task.done():
            self._async_cleanup_task = asyncio.create_task(self._async_cleanup_daemon())
            self.logger.info("Async session cleanup started")
    
    async def _async_cleanup_daemon(self):
        """Async daemon for continuous cleanup and offline queue processing"""
        while not self._cleanup_stop_event.is_set():
            try:
                # Cleanup expired sessions
                self._cleanup_expired_sessions()
                
                # Process offline queue
                await self._process_offline_queue_async()
                
                # Flush batched notifications
                self._flush_pending_notifications()
                
                # Enforce LRU cache size limit
                self._enforce_cache_size_limit()
                
                # Wait for next cleanup cycle
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_stop_event()),
                        timeout=self.cleanup_interval_minutes * 60
                    )
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    pass  # Continue normal cleanup cycle
                    
            except Exception as e:
                self.logger.error(f"Error in async cleanup daemon: {e}")
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_stop_event()),
                        timeout=60  # Wait 1 minute on error
                    )
                    break
                except asyncio.TimeoutError:
                    pass
    
    async def _wait_for_stop_event(self):
        """Async wait for stop event"""
        while not self._cleanup_stop_event.is_set():
            await asyncio.sleep(1)
    
    async def _process_offline_queue_async(self):
        """Process queued events when backend becomes available (async)"""
        if not self._backend_available:
            return
            
        with self._queue_lock:
            processed_events = []
            
            while self._offline_queue:
                event = self._offline_queue.popleft()
                
                try:
                    # Determine endpoint based on event type
                    endpoint_map = {
                        'start': '/conversations/start',
                        'end': '/conversations/end',
                        'failed': '/conversations/failed',
                        'expired': '/conversations/local-expired',
                        'resume': '/conversations/resume',
                        'batch_expired': '/conversations/batch-expired',
                        'evicted': '/conversations/evicted',
                        'message': '/conversations/message',
                        'history': '/conversations/history'
                    }
                    
                    endpoint = endpoint_map.get(event.event_type)
                    if not endpoint:
                        self.logger.error(f"Unknown event type for async replay: {event.event_type}")
                        continue
                    
                    # Replay the event
                    response = await self._make_request_async("POST", endpoint, event.data)
                    
                    if response.success:
                        processed_events.append(event)
                        self.logger.info(f"Successfully replayed {event.event_type} event from {event.timestamp} (async)")
                    else:
                        event.retry_count += 1
                        if event.retry_count <= event.max_retries:
                            # Re-queue for retry
                            self._offline_queue.appendleft(event)
                            self.logger.warning(f"Failed to replay {event.event_type} event (async), retry {event.retry_count}/{event.max_retries}")
                        else:
                            self.logger.error(f"Giving up on {event.event_type} event after {event.max_retries} retries (async)")
                            processed_events.append(event)  # Remove from queue
                        break  # Stop processing on failure
                        
                except Exception as e:
                    self.logger.error(f"Error replaying {event.event_type} event (async): {e}")
                    event.retry_count += 1
                    if event.retry_count <= event.max_retries:
                        self._offline_queue.appendleft(event)
                    break
            
            if processed_events:
                self.logger.info(f"Processed {len(processed_events)} offline events (async)")
    
    def _cleanup_daemon(self):
        """Daemon thread for continuous cleanup and offline queue processing"""
        while not self._cleanup_stop_event.is_set():
            try:
                # Cleanup expired sessions
                self._cleanup_expired_sessions()
                
                # Process offline queue
                self._process_offline_queue()
                
                # Flush batched notifications
                self._flush_pending_notifications()
                
                # Enforce LRU cache size limit
                self._enforce_cache_size_limit()
                
                # Wait for next cleanup cycle
                if self._cleanup_stop_event.wait(timeout=self.cleanup_interval_minutes * 60):
                    break  # Stop event was set
                    
            except Exception as e:
                self.logger.error(f"Error in cleanup daemon: {e}")
                # Continue running even if there's an error
                if self._cleanup_stop_event.wait(timeout=60):  # Wait 1 minute on error
                    break
    
    def _queue_event_offline(self, event_type: str, data: Dict[str, Any]):
        """Queue an event for later replay when backend is unreachable"""
        with self._queue_lock:
            if len(self._offline_queue) >= self.max_offline_queue_size:
                # Remove oldest event to make room
                removed = self._offline_queue.popleft()
                self.logger.warning(f"Offline queue full, removed oldest event: {removed.event_type}")
            
            event = QueuedEvent(
                event_type=event_type,
                timestamp=datetime.now(),
                data=data
            )
            self._offline_queue.append(event)
            
            self.logger.warning(f"Queued {event_type} event offline (queue size: {len(self._offline_queue)})")
    
    def _process_offline_queue(self):
        """Process queued events when backend becomes available"""
        if not self._backend_available:
            return
            
        with self._queue_lock:
            processed_events = []
            
            while self._offline_queue:
                event = self._offline_queue.popleft()
                
                try:
                    # Determine endpoint based on event type
                    endpoint_map = {
                        'start': '/conversations/start',
                        'end': '/conversations/end',
                        'failed': '/conversations/failed',
                        'expired': '/conversations/local-expired',
                        'resume': '/conversations/resume',
                        'message': '/conversations/message',
                        'history': '/conversations/history'
                    }
                    
                    endpoint = endpoint_map.get(event.event_type)
                    if not endpoint:
                        self.logger.error(f"Unknown event type for replay: {event.event_type}")
                        continue
                    
                    # Replay the event
                    response = self._make_request("POST", endpoint, event.data)
                    
                    if response.success:
                        processed_events.append(event)
                        self.logger.info(f"Successfully replayed {event.event_type} event from {event.timestamp}")
                    else:
                        event.retry_count += 1
                        if event.retry_count <= event.max_retries:
                            # Re-queue for retry
                            self._offline_queue.appendleft(event)
                            self.logger.warning(f"Failed to replay {event.event_type} event, retry {event.retry_count}/{event.max_retries}")
                        else:
                            self.logger.error(f"Giving up on {event.event_type} event after {event.max_retries} retries")
                            processed_events.append(event)  # Remove from queue
                        break  # Stop processing on failure
                        
                except Exception as e:
                    self.logger.error(f"Error replaying {event.event_type} event: {e}")
                    event.retry_count += 1
                    if event.retry_count <= event.max_retries:
                        self._offline_queue.appendleft(event)
                    break
            
            if processed_events:
                self.logger.info(f"Processed {len(processed_events)} offline events")
    
    def _enforce_cache_size_limit(self):
        """Enforce LRU cache size limit by evicting oldest sessions"""
        with self._cache_lock:
            while len(self._session_cache) > self.max_cache_size:
                # Remove least recently used (oldest) session
                session_id, session_info = self._session_cache.popitem(last=False)
                
                # Notify backend about eviction with sliding TTL info
                self._notify_backend_session_evicted(session_id, session_info)
                
                self.logger.info(f"Evicted session {session_id} from cache (LRU policy)")
    
    def _notify_backend_session_evicted(self, session_id: str, session_info: SessionInfo):
        """Notify backend about session eviction from LRU cache"""
        try:
            eviction_data = {
                'session_id': session_id,
                'agent_id': session_info.agent_id,
                'run_id': session_info.run_id,
                'eviction_time': datetime.now().isoformat(),
                'eviction_reason': 'lru_cache_limit',
                'session_age_hours': (datetime.now() - session_info.start_time).total_seconds() / 3600,
                'last_access_time': session_info.last_access_time.isoformat() if session_info.last_access_time else None,
                'was_ended': session_info.is_ended,
                'sliding_ttl_remaining': self._calculate_ttl_remaining(session_info)
            }
            
            # Try to send immediately, or queue offline
            if self._backend_available:
                response = self._make_request("POST", "/conversations/evicted", eviction_data)
                if not response.success:
                    self._queue_event_offline('evicted', eviction_data)
            else:
                self._queue_event_offline('evicted', eviction_data)
                
        except Exception as e:
            self.logger.error(f"Error notifying backend about session {session_id} eviction: {e}")
    
    def _calculate_ttl_remaining(self, session_info: SessionInfo) -> float:
        """Calculate remaining TTL for a session in hours"""
        if session_info.last_access_time:
            expiry_time = session_info.last_access_time + timedelta(hours=self.session_ttl_hours)
            remaining = expiry_time - datetime.now()
            return max(0.0, remaining.total_seconds() / 3600)
        return 0.0
    
    def _access_session_lru(self, session_id: str) -> Optional[SessionInfo]:
        """Access session in LRU cache, moving it to end (most recently used)"""
        with self._cache_lock:
            if session_id in self._session_cache:
                # Move to end (most recently used)
                session_info = self._session_cache.pop(session_id)
                self._session_cache[session_id] = session_info
                return session_info
            return None
    
    def _add_session_lru(self, session_id: str, session_info: SessionInfo):
        """Add session to LRU cache"""
        with self._cache_lock:
            # Remove if already exists
            if session_id in self._session_cache:
                del self._session_cache[session_id]
            
            # Add to end (most recently used)
            self._session_cache[session_id] = session_info
            
            # Enforce size limit
            if len(self._session_cache) > self.max_cache_size:
                # Remove least recently used (oldest)
                old_session_id, old_session_info = self._session_cache.popitem(last=False)
                self._notify_backend_session_evicted(old_session_id, old_session_info)
                self.logger.info(f"Evicted session {old_session_id} from cache (LRU policy)")
    
    def _remove_session_lru(self, session_id: str) -> Optional[SessionInfo]:
        """Remove session from LRU cache"""
        with self._cache_lock:
            return self._session_cache.pop(session_id, None)
    
    def _start_cleanup_timer(self):
        """Start the TTL cleanup timer"""
        self._cleanup_timer = threading.Timer(
            self.cleanup_interval_minutes * 60, 
            self._cleanup_expired_sessions
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions and add them to batch notifications"""
        expired_sessions = []
        
        with self._cache_lock:
            current_time = datetime.now()
            
            # Find expired sessions
            for session_id, session_info in list(self._session_cache.items()):
                if session_info.is_expired(self.session_ttl_hours):
                    expired_sessions.append((session_id, session_info))
            
            # Remove expired sessions from cache
            for session_id, session_info in expired_sessions:
                del self._session_cache[session_id]
        
        # Add to batch notifications
        for session_id, session_info in expired_sessions:
            self._add_to_expiry_batch(session_id, session_info)
            self.logger.debug(f"Session {session_id} expired and added to notification batch")
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        # Restart the cleanup timer
        self._start_cleanup_timer()
    
    def _notify_backend_session_expired(self, session_id: str, session_info: SessionInfo):
        """Notify backend about session expiry from local cache (hybrid model)"""
        try:
            # Check if session is also expired in backend (20 hours)
            backend_expiry = session_info.start_time + timedelta(hours=self.backend_ttl_hours)
            backend_expired = datetime.now() > backend_expiry
            
            expiry_data = {
                'session_id': session_id,
                'agent_id': session_info.agent_id,
                'run_id': session_info.run_id,
                'local_expiry_time': datetime.now().isoformat(),
                'local_ttl_hours': self.session_ttl_hours,
                'backend_ttl_hours': self.backend_ttl_hours,
                'backend_expired': backend_expired,
                'session_duration_hours': (datetime.now() - session_info.start_time).total_seconds() / 3600,
                'last_access_time': session_info.last_access_time.isoformat() if session_info.last_access_time else None,
                'was_ended': session_info.is_ended
            }
            
            response = self._make_request("POST", "/conversations/local-expired", expiry_data)
            
            if response.success:
                self.logger.info(f"Notified backend about local expiry of session {session_id}")
            else:
                self.logger.warning(f"Failed to notify backend about session expiry: {response.error}")
                
        except Exception as e:
            self.logger.error(f"Error notifying backend about session {session_id} expiry: {e}")
    
    def set_session_ttl(self, ttl_hours: float):
        """Update the session TTL"""
        self.session_ttl_hours = ttl_hours
        self.logger.info(f"Session TTL updated to {ttl_hours} hours")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session cache statistics with hybrid model information"""
        with self._cache_lock:
            total_sessions = len(self._session_cache)
            expired_count = sum(
                1 for session in self._session_cache.values() 
                if session.is_expired(self.session_ttl_hours)
            )
            
            ended_count = sum(
                1 for session in self._session_cache.values()
                if session.is_ended
            )
            
            # Calculate average time since last access
            now = datetime.now()
            if total_sessions > 0:
                total_idle_time = sum(
                    (now - session.last_access_time).total_seconds() / 3600
                    for session in self._session_cache.values()
                )
                avg_idle_hours = total_idle_time / total_sessions
                
                # Find session with longest idle time
                longest_idle = max(
                    (now - session.last_access_time).total_seconds() / 3600
                    for session in self._session_cache.values()
                ) if total_sessions > 0 else 0.0
                
                # Find session with shortest idle time
                shortest_idle = min(
                    (now - session.last_access_time).total_seconds() / 3600
                    for session in self._session_cache.values()
                ) if total_sessions > 0 else 0.0
                
                # Sessions that would be expired in backend (20 hours)
                backend_expired_count = sum(
                    1 for session in self._session_cache.values()
                    if (now - session.start_time).total_seconds() / 3600 > self.backend_ttl_hours
                )
            else:
                avg_idle_hours = 0.0
                longest_idle = 0.0
                shortest_idle = 0.0
                backend_expired_count = 0
        
        # Get offline queue stats
        with self._queue_lock:
            queue_size = len(self._offline_queue)
            queue_types = {}
            failed_events = 0
            
            for event in self._offline_queue:
                event_type = event.event_type
                queue_types[event_type] = queue_types.get(event_type, 0) + 1
                if event.retry_count > 0:
                    failed_events += 1
        
        # Get pending notification stats
        with self._notification_lock:
            pending_notifications = len(self._pending_expiry_notifications)
        
        # Cache efficiency metrics
        cache_utilization = (total_sessions / self.max_cache_size) * 100 if self.max_cache_size > 0 else 0
        
        return {
            # Session Cache Metrics
            'total_cached_sessions': total_sessions,
            'expired_sessions_pending_cleanup': expired_count,
            'active_sessions': total_sessions - expired_count - ended_count,
            'ended_sessions': ended_count,
            'cache_utilization_percent': round(cache_utilization, 1),
            'max_cache_size': self.max_cache_size,
            
            # TTL Configuration
            'local_ttl_hours': self.session_ttl_hours,
            'backend_ttl_hours': self.backend_ttl_hours,
            'cleanup_interval_minutes': self.cleanup_interval_minutes,
            
            # Hybrid Model Status
            'hybrid_model_enabled': True,
            'sliding_ttl_enabled': True,
            'backend_available': self._backend_available,
            'last_backend_check': self._last_backend_check.isoformat(),
            'backend_expired_sessions': backend_expired_count,
            
            # Session Activity Metrics
            'avg_idle_time_hours': round(avg_idle_hours, 2),
            'longest_idle_time_hours': round(longest_idle, 2),
            'shortest_idle_time_hours': round(shortest_idle, 2),
            
            # Offline Queue Metrics
            'offline_queue_size': queue_size,
            'max_offline_queue_size': self.max_offline_queue_size,
            'offline_queue_utilization_percent': round((queue_size / self.max_offline_queue_size) * 100, 1) if self.max_offline_queue_size > 0 else 0,
            'queued_event_types': queue_types,
            'failed_replay_events': failed_events,
            
            # Notification Batching
            'pending_notifications': pending_notifications,
            'batch_notification_size': self.batch_notification_size,
            
            # System Health
            'cleanup_daemon_running': self._cleanup_thread is not None and self._cleanup_thread.is_alive(),
            'async_cleanup_running': self._async_cleanup_task is not None and not self._async_cleanup_task.done()
        }
    
    def touch_session(self, session_id: str) -> bool:
        """Touch a session to reset its TTL timer (this IS an activity)"""
        session_info = self._access_session_for_activity(session_id)
        if session_info:
            self.logger.debug(f"Session {session_id} TTL timer reset (manual touch)")
            return True
        else:
            self.logger.warning(f"Cannot touch session {session_id}: not found in cache")
            return False
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is active with hybrid cache/backend fallback"""
        # This is a query operation, not an activity - don't reset TTL
        session_info = self._get_session_with_fallback(session_id, is_activity=False)
        
        if session_info:
            # Session exists (either from cache or backend)
            if session_info.is_ended:
                self.logger.debug(f"Session {session_id} has already ended")
                return False
            
            # Check if session is still within local TTL
            is_active = not session_info.is_expired(self.session_ttl_hours)
            if is_active:
                self.logger.debug(f"Session {session_id} is active")
            else:
                self.logger.debug(f"Session {session_id} expired in local cache")
                
                # Remove from local cache but check backend TTL
                self._remove_session_lru(session_id)
                
                # Check if still within backend TTL (20 hours from start)
                backend_expiry = session_info.start_time + timedelta(hours=self.backend_ttl_hours)
                if datetime.now() < backend_expiry:
                    self.logger.info(f"Session {session_id} expired locally but still valid in backend")
                    is_active = True
                else:
                    self.logger.info(f"Session {session_id} expired in both local cache and backend")
                    is_active = False
            
            return is_active
        else:
            # Session not found in cache or backend
            self.logger.debug(f"Session {session_id} not found")
            return False
    
    def get_session_ttl_remaining(self, session_id: str) -> Optional[float]:
        """Get remaining TTL for a session in hours (query operation, no TTL reset)"""
        session_info = self._access_session_for_query(session_id)
        if session_info:
            expiry_time = session_info.last_access_time + timedelta(hours=self.session_ttl_hours)
            remaining = expiry_time - datetime.now()
            remaining_hours = remaining.total_seconds() / 3600
            
            self.logger.debug(f"Session {session_id} TTL remaining: {remaining_hours:.2f} hours")
            return max(0.0, remaining_hours)
        return None

    def _add_to_expiry_batch(self, session_id: str, session_info: SessionInfo):
        """Add session to pending expiry notifications batch"""
        with self._notification_lock:
            # Check if session is also expired in backend (20 hours)
            backend_expiry = session_info.start_time + timedelta(hours=self.backend_ttl_hours)
            backend_expired = datetime.now() > backend_expiry
            
            expiry_data = {
                'session_id': session_id,
                'agent_id': session_info.agent_id,
                'run_id': session_info.run_id,
                'local_expiry_time': datetime.now().isoformat(),
                'local_ttl_hours': self.session_ttl_hours,
                'backend_ttl_hours': self.backend_ttl_hours,
                'backend_expired': backend_expired,
                'session_duration_hours': (datetime.now() - session_info.start_time).total_seconds() / 3600,
                'last_access_time': session_info.last_access_time.isoformat() if session_info.last_access_time else None,
                'was_ended': session_info.is_ended
            }
            
            self._pending_expiry_notifications.append(expiry_data)
            
            # Auto-flush if batch is full
            if len(self._pending_expiry_notifications) >= self.batch_notification_size:
                self._flush_pending_notifications()
    
    def _flush_pending_notifications(self):
        """Flush pending expiry notifications to backend in batch"""
        with self._notification_lock:
            if not self._pending_expiry_notifications:
                return
            
            batch = self._pending_expiry_notifications[:]
            self._pending_expiry_notifications.clear()
        
        # Send batch notification
        try:
            batch_data = {
                'batch_id': f"expiry_batch_{int(datetime.now().timestamp())}",
                'timestamp': datetime.now().isoformat(),
                'expired_sessions': batch,
                'batch_size': len(batch)
            }
            
            if self._backend_available:
                response = self._make_request("POST", "/conversations/batch-expired", batch_data)
                if response.success:
                    self.logger.info(f"Sent batch expiry notification for {len(batch)} sessions")
                else:
                    self.logger.warning(f"Failed to send batch expiry notification: {response.error}")
                    self._queue_event_offline('batch_expired', batch_data)
            else:
                self._queue_event_offline('batch_expired', batch_data)
                
        except Exception as e:
            self.logger.error(f"Error sending batch expiry notification: {e}")
            # Queue the batch for retry
            self._queue_event_offline('batch_expired', batch_data)

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """Make HTTP request with offline queueing support"""
        try:
            response = self.api_client.make_request(method, endpoint, data)
            
            # Update backend availability status
            if response.success:
                if not self._backend_available:
                    self._backend_available = True
                    self.logger.info("Backend connection restored")
            else:
                # Check if it's a connectivity issue
                if any(error in response.error.lower() for error in ['timeout', 'connection', 'unreachable', 'network']):
                    if self._backend_available:
                        self._backend_available = False
                        self.logger.warning("Backend appears to be unreachable")
            
            self._last_backend_check = datetime.now()
            return response
            
        except Exception as e:
            # Backend is likely unreachable
            if self._backend_available:
                self._backend_available = False
                self.logger.warning(f"Backend unreachable, switching to offline mode: {e}")
            
            self._last_backend_check = datetime.now()
            
            # Return failed response
            return APIResponse(
                success=False,
                status_code=0,
                data=None,
                message=f"Backend unreachable: {str(e)}"
            )
    
    async def _make_request_async(self, method: str, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """Make async HTTP request with offline queueing support"""
        try:
            response = await self.api_client.make_request_async(method, endpoint, data)
            
            # Update backend availability status
            if response.success:
                if not self._backend_available:
                    self._backend_available = True
                    self.logger.info("Backend connection restored (async)")
            else:
                # Check if it's a connectivity issue
                if any(error in response.error.lower() for error in ['timeout', 'connection', 'unreachable', 'network']):
                    if self._backend_available:
                        self._backend_available = False
                        self.logger.warning("Backend appears to be unreachable (async)")
            
            self._last_backend_check = datetime.now()
            return response
            
        except Exception as e:
            # Backend is likely unreachable
            if self._backend_available:
                self._backend_available = False
                self.logger.warning(f"Backend unreachable (async), switching to offline mode: {e}")
            
            self._last_backend_check = datetime.now()
            
            # Return failed response
            return APIResponse(
                success=False,
                status_code=0,
                data=None,
                message=f"Backend unreachable: {str(e)}"
            )
    
    async def _parse_async_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Parse async response safely"""
        try:
            return await response.json()
        except aiohttp.ContentTypeError:
            text = await response.text()
            return {'raw_response': text}
    
    def start_conversation(self, agent_id: str, user_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start tracking a new conversation with custom session ID format"""
        start_time = datetime.now()
        session_id = self._generate_session_id(agent_id, start_time)
        
        data = asdict(ConversationStartData(
            agent_id=agent_id,
            session_id=session_id,
            run_id=session_id, # Use session_id as run_id for new sessions
            user_id=user_id,
            start_time=start_time.isoformat(),
            metadata=metadata
        ))
        
        response = self._make_request('POST', '/conversations/start', data)
        
        if response.success:
            # Cache session info for validation
            with self._cache_lock:
                self._add_session_lru(session_id, SessionInfo(
                    agent_id=agent_id,
                    start_time=start_time,
                    run_id=session_id, # Use session_id as run_id for new sessions
                    user_id=user_id
                ))
            
            self.logger.info(f"Conversation {session_id} started for agent {agent_id}")
            return session_id
        else:
            self.logger.error(f"Failed to start conversation: {response.error}")
            return None
    
    async def start_conversation_async(self, agent_id: str, user_id: Optional[str] = None,
                                      metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Async version of start_conversation"""
        start_time = datetime.now()
        session_id = self._generate_session_id(agent_id, start_time)
        
        data = asdict(ConversationStartData(
            agent_id=agent_id,
            session_id=session_id,
            run_id=session_id, # Use session_id as run_id for new sessions
            user_id=user_id,
            start_time=start_time.isoformat(),
            metadata=metadata
        ))
        
        response = await self._make_request_async('POST', '/conversations/start', data)
        
        if response.success:
            # Cache session info for validation
            with self._cache_lock:
                self._add_session_lru(session_id, SessionInfo(
                    agent_id=agent_id,
                    start_time=start_time,
                    run_id=session_id, # Use session_id as run_id for new sessions
                    user_id=user_id
                ))
            
            self.logger.info(f"Conversation {session_id} started for agent {agent_id}")
            return session_id
        else:
            self.logger.error(f"Failed to start conversation: {response.error}")
            return None
    
    def end_conversation(self, session_id: str,
                        quality_score: Optional[Union[int, ConversationQuality]] = None,
                        user_feedback: Optional[str] = None,
                        message_count: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """End a conversation with hybrid session management and seamless resumption"""
        # Use hybrid cache/backend fallback to get session info (this is an activity)
        session_info = self._get_session_with_fallback(session_id, is_activity=True)
        
        if session_info:
            agent_id = session_info.agent_id
            start_time = session_info.start_time
            run_id = session_info.run_id
            
            # Mark session as ended locally
            session_info.is_ended = True
            
            # Remove from cache as conversation is ending
            self._remove_session_lru(session_id)
                    
            self.logger.info(f"Session {session_id} found and ending (agent: {agent_id})")
        else:
            self.logger.warning(f"Session {session_id} not found in cache or backend. Proceeding with fallback parsing.")
            # Try to extract agent_id from session_id format: agent_id_timestamp_random
            try:
                parts = session_id.split('_')
                if len(parts) >= 2:
                    agent_id = parts[0]
                    timestamp = int(parts[1])
                    start_time = datetime.fromtimestamp(timestamp)
                    run_id = session_id  # Use session_id as run_id
                else:
                    self.logger.error(f"Cannot extract agent_id from session_id format: {session_id}")
                    return False
            except (ValueError, IndexError) as e:
                self.logger.error(f"Cannot parse session_id {session_id}: {e}")
                return False
        
        # Calculate duration
        end_time = datetime.now()
        duration_seconds = int((end_time - start_time).total_seconds())
        
        # Convert quality score if needed
        if isinstance(quality_score, int):
            quality_score = ConversationQuality.from_int_safe(quality_score)
        
        quality_value = quality_score.value if quality_score else None
        
        data = asdict(ConversationEndData(
            agent_id=agent_id,
            session_id=session_id,
            run_id=run_id,
            end_time=end_time.isoformat(),
            duration_seconds=duration_seconds,
            quality_score=quality_value,
            user_feedback=user_feedback,
            message_count=message_count,
            metadata=metadata
        ))
        
        # Try to send to backend, queue if offline
        if self._backend_available:
            response = self._make_request("POST", "/conversations/end", data)
            if response.success:
                self.logger.info(f"Successfully ended conversation {session_id}")
                return True
            else:
                self.logger.warning(f"Failed to end conversation: {response.error}")
                self._queue_event_offline('end', data)
                return False
        else:
            self.logger.warning("Backend unavailable, queueing end conversation event")
            self._queue_event_offline('end', data)
            return True  # Return True since we queued it
    
    def record_failed_session(self, 
                             session_id: Optional[str] = None,
                             agent_id: Optional[str] = None,
                             error_message: Optional[str] = None,
                             failure_reason: Optional[str] = None,
                             error_details: Optional[Dict[str, Any]] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record a failed conversation session with hybrid session management
        
        Args:
            session_id: Session ID (if available)
            agent_id: Agent ID (fallback if session_id not available)
            error_message: Error message (legacy parameter name)
            failure_reason: Failure reason (new parameter name)
            error_details: Additional error details
            metadata: Additional metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Handle parameter flexibility for backward compatibility
        actual_error_message = failure_reason or error_message or "Unknown error"
        actual_metadata = metadata or error_details or {}
        
        if session_id:
            # Use hybrid cache/backend fallback to get session info (this is an activity)
            session_info = self._get_session_with_fallback(session_id, is_activity=True)
            
            if session_info:
                actual_agent_id = session_info.agent_id
                start_time = session_info.start_time
                run_id = session_info.run_id
                
                # Mark session as ended and remove from cache
                session_info.is_ended = True
                self._remove_session_lru(session_id)
            else:
                # Session not found, try to extract from session_id format or use fallback
                actual_agent_id = agent_id or self._extract_agent_id_from_session(session_id)
                start_time = datetime.now()
                run_id = "unknown"
        elif agent_id:
            # Legacy mode: agent_id provided directly
            actual_agent_id = agent_id
            session_id = f"failed_{actual_agent_id}_{int(datetime.now().timestamp())}"
            start_time = datetime.now()
            run_id = "legacy"
        else:
            self.logger.error("Either session_id or agent_id must be provided")
            return False

        # Prepare failed session data
        failed_session_data = {
            "session_id": session_id,
            "agent_id": actual_agent_id,
            "timestamp": datetime.now().isoformat(),
            "client_id": self.client_id,
            "failure_reason": actual_error_message,
            "error_details": actual_metadata,
            "start_time": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
            "run_id": run_id
        }
        
        # Send to backend
        response = self._make_request('POST', '/conversations/failed-session', failed_session_data)
        
        if response.success:
            self.logger.info(f"Failed session recorded for agent {actual_agent_id}: {actual_error_message}")
            
            # Update internal metrics
            with self._metrics_lock:
                self._performance_metrics.failed_sessions += 1
                if actual_agent_id not in self._performance_metrics.agent_failure_counts:
                    self._performance_metrics.agent_failure_counts[actual_agent_id] = 0
                self._performance_metrics.agent_failure_counts[actual_agent_id] += 1
                
            return True
        else:
            self.logger.error(f"Failed to record failed session: {response.error}")
            self._queue_event_offline('record_failed_session', failed_session_data)
            return False
    
    def _extract_agent_id_from_session(self, session_id: str) -> str:
        """Extract agent_id from session_id format (if follows convention)"""
        try:
            # Try to extract from format like "session_agentid_timestamp" 
            parts = session_id.split('_')
            if len(parts) >= 3 and parts[0] == 'session':
                return parts[1]
            # Try other common formats
            if 'agent' in session_id:
                import re
                match = re.search(r'agent[_-]?([a-zA-Z0-9]+)', session_id)
                if match:
                    return match.group(1)
        except Exception:
            pass
        return "unknown"
    
    def get_success_rates(self, agent_id: Optional[str] = None, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get success rate metrics for agents"""
        params = {}
        if agent_id:
            params['agent_id'] = agent_id
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f'/metrics/success-rates?{query_string}' if query_string else '/metrics/success-rates'
        
        response = self._make_request('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get success rates: {response.error}")
            return None
    
    async def get_success_rates_async(self, agent_id: Optional[str] = None,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Async version of get_success_rates"""
        params = {}
        if agent_id:
            params['agent_id'] = agent_id
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f'/metrics/success-rates?{query_string}' if query_string else '/metrics/success-rates'
        
        response = await self._make_request_async('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get success rates: {response.error}")
            return None
    
    def get_response_times(self, agent_id: Optional[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get response time metrics for agents"""
        params = {}
        if agent_id:
            params['agent_id'] = agent_id
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f'/metrics/response-times?{query_string}' if query_string else '/metrics/response-times'
        
        response = self._make_request('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get response times: {response.error}")
            return None
    
    async def get_response_times_async(self, agent_id: Optional[str] = None,
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Async version of get_response_times"""
        params = {}
        if agent_id:
            params['agent_id'] = agent_id
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f'/metrics/response-times?{query_string}' if query_string else '/metrics/response-times'
        
        response = await self._make_request_async('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get response times: {response.error}")
            return None
    
    def get_conversation_quality(self, agent_id: Optional[str] = None,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get conversation quality metrics for agents"""
        params = {}
        if agent_id:
            params['agent_id'] = agent_id
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f'/metrics/quality?{query_string}' if query_string else '/metrics/quality'
        
        response = self._make_request('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get conversation quality: {response.error}")
            return None
    
    async def get_conversation_quality_async(self, agent_id: Optional[str] = None,
                                           start_date: Optional[str] = None,
                                           end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Async version of get_conversation_quality"""
        params = {}
        if agent_id:
            params['agent_id'] = agent_id
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f'/metrics/quality?{query_string}' if query_string else '/metrics/quality'
        
        response = await self._make_request_async('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get conversation quality: {response.error}")
            return None
    
    def get_failed_sessions(self, agent_id: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get failed session metrics for agents"""
        params = {}
        if agent_id:
            params['agent_id'] = agent_id
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f'/metrics/failures?{query_string}' if query_string else '/metrics/failures'
        
        response = self._make_request('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get failed sessions: {response.error}")
            return None
    
    async def get_failed_sessions_async(self, agent_id: Optional[str] = None,
                                       start_date: Optional[str] = None,
                                       end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Async version of get_failed_sessions"""
        params = {}
        if agent_id:
            params['agent_id'] = agent_id
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f'/metrics/failures?{query_string}' if query_string else '/metrics/failures'
        
        response = await self._make_request_async('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get failed sessions: {response.error}")
            return None
    
    def get_performance_overview(self, agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get comprehensive performance overview"""
        endpoint = '/metrics/overview'
        if agent_id:
            endpoint = f'/metrics/overview?agent_id={agent_id}'
        
        response = self._make_request('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get performance overview: {response.error}")
            return None
    
    async def get_performance_overview_async(self, agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Async version of get_performance_overview"""
        endpoint = '/metrics/overview'
        if agent_id:
            endpoint = f'/metrics/overview?agent_id={agent_id}'
        
        response = await self._make_request_async('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get performance overview: {response.error}")
            return None
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information from backend"""
        response = self._make_request('GET', f'/conversations/{session_id}')
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get session info for {session_id}: {response.error}")
            return None
    
    async def get_session_info_async(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Async version of get_session_info"""
        response = await self._make_request_async('GET', f'/conversations/{session_id}')
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get session info for {session_id}: {response.error}")
            return None
    
    def get_active_conversations(self, agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get active conversations from backend"""
        endpoint = '/conversations/active'
        if agent_id:
            endpoint = f'/conversations/active?agent_id={agent_id}'
        
        response = self._make_request('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get active conversations: {response.error}")
            return None
    
    async def get_active_conversations_async(self, agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Async version of get_active_conversations"""
        endpoint = '/conversations/active'
        if agent_id:
            endpoint = f'/conversations/active?agent_id={agent_id}'
        
        response = await self._make_request_async('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get active conversations: {response.error}")
            return None
    
    def _retrieve_session_from_backend(self, session_id: str) -> Optional[SessionInfo]:
        """Retrieve session information from backend when local cache expires"""
        try:
            endpoint = f"/conversations/{session_id}"
            response = self._make_request("GET", endpoint)
            
            if response.success and response.data:
                session_data = response.data
                
                # Validate session data
                if not self._validate_session_data(session_data):
                    self.logger.error(f"Invalid session data from backend for {session_id}")
                    return self._create_fallback_session_info(session_id)
                
                # Reconstruct SessionInfo from backend data
                session_info = SessionInfo(
                    agent_id=session_data.get('agent_id'),
                    start_time=datetime.fromisoformat(session_data.get('start_time')),
                    run_id=session_data.get('run_id', session_id),
                    user_id=session_data.get('user_id'),
                    is_ended=session_data.get('is_ended', False)
                )
                
                # Touch the session since we're accessing it
                session_info.touch()
                
                # Add back to local cache for future quick access
                with self._cache_lock:
                    self._add_session_lru(session_id, session_info)
                
                self.logger.info(f"Retrieved session {session_id} from backend and restored to cache")
                return session_info
            else:
                self.logger.warning(f"Session {session_id} not found in backend: {response.error}")
                return self._create_fallback_session_info(session_id)
                
        except Exception as e:
            self._handle_session_error(session_id, "backend_retrieval", e)
            return self._create_fallback_session_info(session_id)
    
    async def _retrieve_session_from_backend_async(self, session_id: str) -> Optional[SessionInfo]:
        """Retrieve session information from backend when local cache expires (async)"""
        try:
            endpoint = f"/conversations/{session_id}"
            response = await self._make_request_async("GET", endpoint)
            
            if response.success and response.data:
                session_data = response.data
                
                # Validate session data
                if not self._validate_session_data(session_data):
                    self.logger.error(f"Invalid session data from backend for {session_id}")
                    return self._create_fallback_session_info(session_id)
                
                # Reconstruct SessionInfo from backend data
                session_info = SessionInfo(
                    agent_id=session_data.get('agent_id'),
                    start_time=datetime.fromisoformat(session_data.get('start_time')),
                    run_id=session_data.get('run_id', session_id),
                    user_id=session_data.get('user_id'),
                    is_ended=session_data.get('is_ended', False)
                )
                
                # Touch the session since we're accessing it
                session_info.touch()
                
                # Add back to local cache for future quick access
                with self._cache_lock:
                    self._add_session_lru(session_id, session_info)
                
                self.logger.info(f"Retrieved session {session_id} from backend and restored to cache")
                return session_info
            else:
                self.logger.warning(f"Session {session_id} not found in backend: {response.error}")
                return self._create_fallback_session_info(session_id)
                
        except Exception as e:
            self._handle_session_error(session_id, "async_backend_retrieval", e)
            return self._create_fallback_session_info(session_id)
    
    def _get_session_with_fallback(self, session_id: str, is_activity: bool = True) -> Optional[SessionInfo]:
        """Get session with hybrid cache/backend fallback logic"""
        # First, check local cache
        session_info = None
        if is_activity:
            session_info = self._access_session_for_activity(session_id)
        else:
            session_info = self._access_session_for_query(session_id)
        
        if session_info:
            # Check if expired locally
            if session_info.is_expired(self.session_ttl_hours):
                self.logger.info(f"Session {session_id} expired in local cache, checking backend...")
                # Remove from local cache
                self._remove_session_lru(session_id)
                session_info = None
            else:
                # Still valid in local cache
                activity_msg = "activity" if is_activity else "query"
                self.logger.debug(f"Session {session_id} found in cache for {activity_msg}")
                return session_info
        
        # Session not in cache or expired locally, try backend
        self.logger.info(f"Session {session_id} not in local cache, attempting backend retrieval...")
        return self._retrieve_session_from_backend(session_id)
    
    async def _get_session_with_fallback_async(self, session_id: str, is_activity: bool = True) -> Optional[SessionInfo]:
        """Get session with hybrid cache/backend fallback logic (async)"""
        # First, check local cache
        session_info = None
        if is_activity:
            session_info = self._access_session_for_activity(session_id)
        else:
            session_info = self._access_session_for_query(session_id)
        
        if session_info:
            # Check if expired locally
            if session_info.is_expired(self.session_ttl_hours):
                self.logger.info(f"Session {session_id} expired in local cache, checking backend...")
                # Remove from local cache
                self._remove_session_lru(session_id)
                session_info = None
            else:
                # Still valid in local cache
                activity_msg = "activity" if is_activity else "query"
                self.logger.debug(f"Session {session_id} found in cache for {activity_msg}")
                return session_info
        
        # Session not in cache or expired locally, try backend
        self.logger.info(f"Session {session_id} not in local cache, attempting backend retrieval...")
        return await self._retrieve_session_from_backend_async(session_id)

    def close(self):
        """Close resources and stop cleanup daemon"""
        self.logger.info("Shutting down AgentPerformanceTracker...")
        
        # Stop cleanup daemon
        self._cleanup_stop_event.set()
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
            if self._cleanup_thread.is_alive():
                self.logger.warning("Cleanup thread did not stop gracefully")
        
        # Flush any pending notifications
        self._flush_pending_notifications()
        
        # Process remaining offline queue (best effort)
        try:
            self._process_offline_queue()
        except Exception as e:
            self.logger.error(f"Error processing final offline queue: {e}")
        
        # Close API client
        if hasattr(self.api_client, 'close'):
            self.api_client.close()
        
        # Clear caches
        with self._cache_lock:
            self._session_cache.clear()
        
        with self._queue_lock:
            self._offline_queue.clear()
        
        with self._notification_lock:
            self._pending_expiry_notifications.clear()
        
        self.logger.info("AgentPerformanceTracker shutdown complete")

    async def close_async(self):
        """Close resources asynchronously and stop async cleanup"""
        self.logger.info("Shutting down AgentPerformanceTracker (async)...")
        
        # Stop cleanup tasks
        self._cleanup_stop_event.set()
        
        if self._async_cleanup_task and not self._async_cleanup_task.done():
            try:
                await asyncio.wait_for(self._async_cleanup_task, timeout=5)
            except asyncio.TimeoutError:
                self.logger.warning("Async cleanup task did not stop gracefully")
                self._async_cleanup_task.cancel()
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
            if self._cleanup_thread.is_alive():
                self.logger.warning("Cleanup thread did not stop gracefully")
        
        # Flush any pending notifications
        self._flush_pending_notifications()
        
        # Process remaining offline queue (best effort)
        try:
            await self._process_offline_queue_async()
        except Exception as e:
            self.logger.error(f"Error processing final offline queue (async): {e}")
        
        # Close API client
        if hasattr(self.api_client, 'close_async'):
            await self.api_client.close_async()
        elif hasattr(self.api_client, 'close'):
            self.api_client.close()
        
        # Clear caches
        with self._cache_lock:
            self._session_cache.clear()
        
        with self._queue_lock:
            self._offline_queue.clear()
        
        with self._notification_lock:
            self._pending_expiry_notifications.clear()
        
        self.logger.info("AgentPerformanceTracker shutdown complete (async)")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_async()

    def resume_conversation(self, session_id: str, agent_id: str, 
                           user_id: Optional[str] = None,
                           context: Optional[Dict[str, Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Resume a conversation that was retrieved from backend"""
        try:
            start_time = datetime.now()
            
            # Create session info for resumed conversation
            session_info = SessionInfo(
                agent_id=agent_id,
                start_time=start_time,
                run_id=session_id,  # Use existing session_id as run_id
                user_id=user_id,
                is_ended=False
            )
            
            # Add to local cache
            with self._cache_lock:
                self._add_session_lru(session_id, session_info)
            
            # Send resumption to backend
            data = asdict(ConversationStartData(
                agent_id=agent_id,
                session_id=session_id,
                run_id=session_id,
                user_id=user_id,
                start_time=start_time.isoformat(),
                context=context,  # Include context for resumption
                metadata=metadata
            ))
            
            response = self._make_request("POST", "/conversations/resume", data)
            
            if response.success:
                self.logger.info(f"Successfully resumed conversation {session_id} for agent {agent_id}")
                return True
            else:
                self.logger.error(f"Failed to resume conversation: {response.error}")
                # Remove from cache if backend failed
                with self._cache_lock:
                    if session_id in self._session_cache:
                        del self._session_cache[session_id]
                return False
                
        except Exception as e:
            self.logger.error(f"Error resuming conversation {session_id}: {e}")
            return False

    async def resume_conversation_async(self, session_id: str, agent_id: str, 
                                       user_id: Optional[str] = None,
                                       context: Optional[Dict[str, Any]] = None,
                                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Resume a conversation that was retrieved from backend (async)"""
        try:
            start_time = datetime.now()
            
            # Create session info for resumed conversation
            session_info = SessionInfo(
                agent_id=agent_id,
                start_time=start_time,
                run_id=session_id,  # Use existing session_id as run_id
                user_id=user_id,
                is_ended=False
            )
            
            # Add to local cache
            with self._cache_lock:
                self._add_session_lru(session_id, session_info)
            
            # Send resumption to backend
            data = asdict(ConversationStartData(
                agent_id=agent_id,
                session_id=session_id,
                run_id=session_id,
                user_id=user_id,
                start_time=start_time.isoformat(),
                context=context,  # Include context for resumption
                metadata=metadata
            ))
            
            response = await self._make_request_async("POST", "/conversations/resume", data)
            
            if response.success:
                self.logger.info(f"Successfully resumed conversation {session_id} for agent {agent_id}")
                return True
            else:
                self.logger.error(f"Failed to resume conversation: {response.error}")
                # Remove from cache if backend failed
                with self._cache_lock:
                    if session_id in self._session_cache:
                        del self._session_cache[session_id]
                return False
                
        except Exception as e:
            self.logger.error(f"Error resuming conversation {session_id}: {e}")
            return False

    def _handle_session_error(self, session_id: str, operation: str, error: Exception) -> bool:
        """Handle session-related errors with appropriate fallbacks"""
        error_msg = str(error)
        
        if "404" in error_msg or "not found" in error_msg.lower():
            self.logger.warning(f"Session {session_id} not found during {operation} - may have expired in backend")
            # Remove from local cache if exists
            with self._cache_lock:
                if session_id in self._session_cache:
                    del self._session_cache[session_id]
            return False
            
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            self.logger.error(f"Network error during {operation} for session {session_id}: {error}")
            # Keep local cache but log the issue
            return False
            
        elif "403" in error_msg or "unauthorized" in error_msg.lower():
            self.logger.error(f"Authorization error during {operation} for session {session_id}: {error}")
            return False
            
        else:
            self.logger.error(f"Unexpected error during {operation} for session {session_id}: {error}")
            return False
    
    def _validate_session_data(self, session_data: Dict[str, Any]) -> bool:
        """Validate session data retrieved from backend"""
        required_fields = ['agent_id', 'start_time']
        
        for field in required_fields:
            if field not in session_data or session_data[field] is None:
                self.logger.error(f"Missing required field '{field}' in session data")
                return False
        
        # Validate datetime format
        try:
            datetime.fromisoformat(session_data['start_time'])
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid start_time format in session data: {e}")
            return False
            
        return True
    
    def _create_fallback_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Create fallback session info when backend retrieval fails"""
        try:
            # Try to extract info from session_id format: agent_id_timestamp_random
            parts = session_id.split('_')
            if len(parts) >= 2:
                agent_id = parts[0]
                timestamp = int(parts[1])
                start_time = datetime.fromtimestamp(timestamp)
                
                # Create minimal session info
                session_info = SessionInfo(
                    agent_id=agent_id,
                    start_time=start_time,
                    run_id=session_id,
                    user_id=None,
                    is_ended=False
                )
                
                self.logger.info(f"Created fallback session info for {session_id} (agent: {agent_id})")
                return session_info
            else:
                self.logger.error(f"Cannot create fallback for session_id format: {session_id}")
                return None
                
        except (ValueError, IndexError) as e:
            self.logger.error(f"Cannot parse session_id {session_id} for fallback: {e}")
            return None

    def _access_session_for_query(self, session_id: str) -> Optional[SessionInfo]:
        """Access session for read-only queries (does NOT reset TTL)"""
        with self._cache_lock:
            if session_id in self._session_cache:
                # Move to end (most recently used) but don't touch TTL
                session_info = self._session_cache.pop(session_id)
                self._session_cache[session_id] = session_info
                return session_info
            return None
    
    def _access_session_for_activity(self, session_id: str) -> Optional[SessionInfo]:
        """Access session for activity (DOES reset TTL)"""
        with self._cache_lock:
            if session_id in self._session_cache:
                # Move to end (most recently used) AND reset TTL
                session_info = self._session_cache.pop(session_id)
                session_info.touch()  # Reset sliding TTL
                self._session_cache[session_id] = session_info
                self.logger.debug(f"Session {session_id} activity - TTL reset")
                return session_info
            return None

    def log_message(self, session_id: str, message_type: str, content: str,
                   response_time_ms: Optional[int] = None,
                   tokens_used: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log an individual message within a conversation
        
        Args:
            session_id: The conversation session ID
            message_type: Type of message ("user", "agent", "system")
            content: The message content
            response_time_ms: Response time in milliseconds (for agent messages)
            tokens_used: Number of tokens used (for LLM messages)
            metadata: Additional metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get session info to extract agent_id
        session_info = self._get_session_with_fallback(session_id, is_activity=True)
        
        if session_info:
            agent_id = session_info.agent_id
        else:
            # Try to extract agent_id from session_id format
            agent_id = self._extract_agent_id_from_session(session_id)
            self.logger.warning(f"Session {session_id} not found, using extracted agent_id: {agent_id}")
        
        # Generate unique message ID
        message_id = f"{session_id}_{message_type}_{int(datetime.now().timestamp()*1000)}"
        
        # Prepare message data
        message_data = asdict(MessageData(
            session_id=session_id,
            agent_id=agent_id,
            message_id=message_id,
            timestamp=datetime.now().isoformat(),
            message_type=message_type,
            content=content,
            metadata=metadata,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used
        ))
        
        # Send to backend
        response = self._make_request('POST', '/conversations/message', message_data)
        
        if response.success:
            self.logger.info(f"Message logged for session {session_id} (type: {message_type})")
            return True
        else:
            self.logger.error(f"Failed to log message: {response.error}")
            # Queue for offline replay
            self._queue_event_offline('message', message_data)
            return False
    
    async def log_message_async(self, session_id: str, message_type: str, content: str,
                               response_time_ms: Optional[int] = None,
                               tokens_used: Optional[int] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async version of log_message"""
        # Get session info to extract agent_id
        session_info = await self._get_session_with_fallback_async(session_id, is_activity=True)
        
        if session_info:
            agent_id = session_info.agent_id
        else:
            # Try to extract agent_id from session_id format
            agent_id = self._extract_agent_id_from_session(session_id)
            self.logger.warning(f"Session {session_id} not found, using extracted agent_id: {agent_id}")
        
        # Generate unique message ID
        message_id = f"{session_id}_{message_type}_{int(datetime.now().timestamp()*1000)}"
        
        # Prepare message data
        message_data = asdict(MessageData(
            session_id=session_id,
            agent_id=agent_id,
            message_id=message_id,
            timestamp=datetime.now().isoformat(),
            message_type=message_type,
            content=content,
            metadata=metadata,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used
        ))
        
        # Send to backend
        response = await self._make_request_async('POST', '/conversations/message', message_data)
        
        if response.success:
            self.logger.info(f"Message logged for session {session_id} (type: {message_type})")
            return True
        else:
            self.logger.error(f"Failed to log message: {response.error}")
            # Queue for offline replay
            self._queue_event_offline('message', message_data)
            return False
    
    def update_conversation_history(self, session_id: str, messages: List[Dict[str, Any]],
                                   conversation_summary: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the complete conversation history for a session
        
        Args:
            session_id: The conversation session ID
            messages: List of message dictionaries with structure:
                     [{"type": "user", "content": "...", "timestamp": "...", "tokens": 10}, ...]
            conversation_summary: Optional summary of the conversation
            metadata: Additional metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get session info to extract agent_id
        session_info = self._get_session_with_fallback(session_id, is_activity=True)
        
        if session_info:
            agent_id = session_info.agent_id
            start_time = session_info.start_time.isoformat()
        else:
            # Try to extract agent_id from session_id format
            agent_id = self._extract_agent_id_from_session(session_id)
            start_time = datetime.now().isoformat()
            self.logger.warning(f"Session {session_id} not found, using extracted agent_id: {agent_id}")
        
        # Calculate totals
        total_messages = len(messages)
        total_tokens = sum(msg.get('tokens', 0) for msg in messages if isinstance(msg.get('tokens'), (int, float)))
        
        # Prepare conversation history data
        history_data = asdict(ConversationHistoryData(
            session_id=session_id,
            agent_id=agent_id,
            messages=messages,
            conversation_summary=conversation_summary,
            total_messages=total_messages,
            total_tokens=total_tokens,
            start_time=start_time,
            last_update=datetime.now().isoformat(),
            metadata=metadata
        ))
        
        # Send to backend
        response = self._make_request('POST', '/conversations/history', history_data)
        
        if response.success:
            self.logger.info(f"Conversation history updated for session {session_id} ({total_messages} messages, {total_tokens} tokens)")
            return True
        else:
            self.logger.error(f"Failed to update conversation history: {response.error}")
            # Queue for offline replay
            self._queue_event_offline('history', history_data)
            return False
    
    async def update_conversation_history_async(self, session_id: str, messages: List[Dict[str, Any]],
                                               conversation_summary: Optional[str] = None,
                                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async version of update_conversation_history"""
        # Get session info to extract agent_id
        session_info = await self._get_session_with_fallback_async(session_id, is_activity=True)
        
        if session_info:
            agent_id = session_info.agent_id
            start_time = session_info.start_time.isoformat()
        else:
            # Try to extract agent_id from session_id format
            agent_id = self._extract_agent_id_from_session(session_id)
            start_time = datetime.now().isoformat()
            self.logger.warning(f"Session {session_id} not found, using extracted agent_id: {agent_id}")
        
        # Calculate totals
        total_messages = len(messages)
        total_tokens = sum(msg.get('tokens', 0) for msg in messages if isinstance(msg.get('tokens'), (int, float)))
        
        # Prepare conversation history data
        history_data = asdict(ConversationHistoryData(
            session_id=session_id,
            agent_id=agent_id,
            messages=messages,
            conversation_summary=conversation_summary,
            total_messages=total_messages,
            total_tokens=total_tokens,
            start_time=start_time,
            last_update=datetime.now().isoformat(),
            metadata=metadata
        ))
        
        # Send to backend
        response = await self._make_request_async('POST', '/conversations/history', history_data)
        
        if response.success:
            self.logger.info(f"Conversation history updated for session {session_id} ({total_messages} messages, {total_tokens} tokens)")
            return True
        else:
            self.logger.error(f"Failed to update conversation history: {response.error}")
            # Queue for offline replay
            self._queue_event_offline('history', history_data)
            return False
    
    def get_conversation_history(self, session_id: str, 
                                include_content: bool = True,
                                limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve conversation history from backend
        
        Args:
            session_id: The conversation session ID
            include_content: Whether to include message content
            limit: Maximum number of messages to retrieve
            
        Returns:
            Dict containing conversation history or None if failed
        """
        params = {}
        if not include_content:
            params['include_content'] = 'false'
        if limit:
            params['limit'] = str(limit)
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f'/conversations/{session_id}/history'
        if query_string:
            endpoint += f'?{query_string}'
        
        response = self._make_request('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get conversation history for {session_id}: {response.error}")
            return None
    
    async def get_conversation_history_async(self, session_id: str, 
                                           include_content: bool = True,
                                           limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Async version of get_conversation_history"""
        params = {}
        if not include_content:
            params['include_content'] = 'false'
        if limit:
            params['limit'] = str(limit)
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f'/conversations/{session_id}/history'
        if query_string:
            endpoint += f'?{query_string}'
        
        response = await self._make_request_async('GET', endpoint)
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get conversation history for {session_id}: {response.error}")
            return None
    
    def log_user_message(self, session_id: str, content: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Convenience method to log a user message"""
        return self.log_message(session_id, "user", content, metadata=metadata)
    
    def log_agent_message(self, session_id: str, content: str, 
                         response_time_ms: Optional[int] = None,
                         tokens_used: Optional[int] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Convenience method to log an agent message"""
        return self.log_message(session_id, "agent", content, 
                               response_time_ms=response_time_ms,
                               tokens_used=tokens_used, 
                               metadata=metadata)
    
    def log_system_message(self, session_id: str, content: str,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Convenience method to log a system message"""
        return self.log_message(session_id, "system", content, metadata=metadata)
    
    async def log_user_message_async(self, session_id: str, content: str, 
                                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async convenience method to log a user message"""
        return await self.log_message_async(session_id, "user", content, metadata=metadata)
    
    async def log_agent_message_async(self, session_id: str, content: str, 
                                     response_time_ms: Optional[int] = None,
                                     tokens_used: Optional[int] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async convenience method to log an agent message"""
        return await self.log_message_async(session_id, "agent", content, 
                                           response_time_ms=response_time_ms,
                                           tokens_used=tokens_used, 
                                           metadata=metadata)
    
    async def log_system_message_async(self, session_id: str, content: str,
                                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async convenience method to log a system message"""
        return await self.log_message_async(session_id, "system", content, metadata=metadata)
