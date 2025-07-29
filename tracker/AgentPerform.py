import time
import uuid
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import aiohttp
from urllib.parse import urljoin
import re

# Import secure components from AgentOper
from .AgentOper import SecureLogger, SecureAPIClient, APIResponse


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
    """Lightweight session information for validation"""
    agent_id: str
    start_time: datetime
    user_id: Optional[str] = None
    last_access_time: Optional[datetime] = None  # For sliding TTL
    
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


@dataclass
class ConversationStartData:
    """Data structure for conversation start"""
    session_id: str
    agent_id: str
    start_time: str
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationEndData:
    """Data structure for conversation end"""
    session_id: str
    agent_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    status: str  # "completed" or "failed"
    quality_score: Optional[int] = None
    user_feedback: Optional[str] = None
    error_message: Optional[str] = None
    message_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetricsQuery:
    """Data structure for performance metrics queries"""
    agent_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    metric_type: Optional[str] = None  # "success_rate", "response_time", "quality", "failures"


class AgentPerformanceTracker:
    """
    Agent Performance Tracker - Handles performance metrics:
    - Success rates
    - Response time
    - Conversation quality
    - Failed sessions
    
    Now includes lightweight session tracking with TTL-based cleanup.
    """
    
    def __init__(self, 
                 base_url: str,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_async: bool = False,
                 session_ttl_hours: float = 10.0,
                 cleanup_interval_minutes: int = 30,
                 logger: Optional[logging.Logger] = None):
        """Initialize the Agent Performance Tracker"""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_async = enable_async
        self.session_ttl_hours = session_ttl_hours
        self.cleanup_interval_minutes = cleanup_interval_minutes
        
        # Setup secure logging
        self.logger = logger or logging.getLogger(__name__)
        self.secure_logger = SecureLogger()
        
        # Initialize secure API client
        self.api_client = SecureAPIClient(base_url, api_key)
        
        # Lightweight session tracking with TTL
        self._session_cache: Dict[str, SessionInfo] = {}
        self._cache_lock = threading.RLock()
        self._cleanup_timer: Optional[threading.Timer] = None
        
        # Start TTL cleanup timer
        self._start_cleanup_timer()
        
        # Log initialization without exposing API key
        self.logger.info(
            self.secure_logger.format_log(
                "AgentPerformanceTracker initialized with base URL: %s, Sliding TTL: %.1f hours",
                self.base_url, self.session_ttl_hours
            )
        )
    
    def _generate_session_id(self, agent_id: str, start_time: datetime) -> str:
        """Generate session ID in format: {agent_id}_{start_time_timestamp}_{random}"""
        timestamp = int(start_time.timestamp())
        random_suffix = str(uuid.uuid4())[:8]  # Short random identifier
        return f"{agent_id}_{timestamp}_{random_suffix}"
    
    def _start_cleanup_timer(self):
        """Start the TTL cleanup timer"""
        self._cleanup_timer = threading.Timer(
            self.cleanup_interval_minutes * 60, 
            self._cleanup_expired_sessions
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions from cache and notify backend"""
        with self._cache_lock:
            expired_sessions = []
            current_time = datetime.now()
            
            # Find expired sessions
            for session_id, session_info in list(self._session_cache.items()):
                if session_info.is_expired(self.session_ttl_hours):
                    expired_sessions.append((session_id, session_info))
                    del self._session_cache[session_id]
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                # Notify backend about expired sessions
                for session_id, session_info in expired_sessions:
                    self._notify_backend_session_expired(session_id, session_info)
        
        # Restart the cleanup timer
        self._start_cleanup_timer()
    
    def _notify_backend_session_expired(self, session_id: str, session_info: SessionInfo):
        """Notify backend about expired session"""
        try:
            data = {
                'session_id': session_id,
                'agent_id': session_info.agent_id,
                'start_time': session_info.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'status': 'expired',
                'duration_seconds': (datetime.now() - session_info.start_time).total_seconds(),
                'expiry_reason': 'ttl_timeout'
            }
            
            response = self._make_request('POST', '/conversations/expired', data)
            
            if response.success:
                self.logger.debug(f"Notified backend about expired session: {session_id}")
            else:
                self.logger.warning(f"Failed to notify backend about expired session {session_id}: {response.error}")
                
        except Exception as e:
            self.logger.error(f"Error notifying backend about expired session {session_id}: {e}")
    
    def set_session_ttl(self, ttl_hours: float):
        """Update the session TTL"""
        self.session_ttl_hours = ttl_hours
        self.logger.info(f"Session TTL updated to {ttl_hours} hours")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session cache statistics with sliding TTL info"""
        with self._cache_lock:
            total_sessions = len(self._session_cache)
            expired_count = sum(
                1 for session in self._session_cache.values() 
                if session.is_expired(self.session_ttl_hours)
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
            else:
                avg_idle_hours = 0.0
                longest_idle = 0.0
                shortest_idle = 0.0
            
            return {
                'total_cached_sessions': total_sessions,
                'expired_sessions_pending_cleanup': expired_count,
                'active_sessions': total_sessions - expired_count,
                'ttl_hours': self.session_ttl_hours,
                'cleanup_interval_minutes': self.cleanup_interval_minutes,
                'sliding_ttl_enabled': True,
                'avg_idle_time_hours': round(avg_idle_hours, 2),
                'longest_idle_time_hours': round(longest_idle, 2),
                'shortest_idle_time_hours': round(shortest_idle, 2)
            }
    
    def touch_session(self, session_id: str) -> bool:
        """Touch a session to reset its TTL timer (sliding TTL)"""
        with self._cache_lock:
            if session_id in self._session_cache:
                session_info = self._session_cache[session_id]
                session_info.touch()
                self.logger.debug(f"Session {session_id} TTL timer reset (sliding TTL)")
                return True
            else:
                self.logger.warning(f"Cannot touch session {session_id}: not found in cache")
                return False
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is active (not expired)"""
        with self._cache_lock:
            if session_id in self._session_cache:
                session_info = self._session_cache[session_id]
                is_active = not session_info.is_expired(self.session_ttl_hours)
                if is_active:
                    # Touch session on access (sliding TTL)
                    session_info.touch()
                    self.logger.debug(f"Session {session_id} checked and TTL reset")
                return is_active
            return False
    
    def get_session_ttl_remaining(self, session_id: str) -> Optional[float]:
        """Get remaining TTL for a session in hours"""
        with self._cache_lock:
            if session_id in self._session_cache:
                session_info = self._session_cache[session_id]
                # Touch session on access (sliding TTL)
                session_info.touch()
                
                expiry_time = session_info.last_access_time + timedelta(hours=self.session_ttl_hours)
                remaining = expiry_time - datetime.now()
                remaining_hours = remaining.total_seconds() / 3600
                
                self.logger.debug(f"Session {session_id} TTL remaining: {remaining_hours:.2f} hours")
                return max(0.0, remaining_hours)
            return None

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """Make HTTP request with secure handling"""
        url = urljoin(self.base_url, endpoint)
        
        # Mask sensitive data in logs
        log_data = self.secure_logger.mask_sensitive_data(data) if data else None
        self.logger.debug(f"Making {method} request to {endpoint} with data: {log_data}")
        
        for attempt in range(self.max_retries + 1):
            try:
                session = self.api_client.get_session()
                
                if method.upper() == 'GET':
                    response = session.get(url, timeout=self.timeout)
                elif method.upper() == 'POST':
                    response = session.post(url, json=data, timeout=self.timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Mask sensitive data in response logging
                if response.status_code < 400:
                    try:
                        response_data = response.json() if response.content else {}
                        masked_response = self.secure_logger.mask_sensitive_data(response_data)
                        self.logger.debug(f"Received response: {masked_response}")
                    except json.JSONDecodeError:
                        self.logger.debug("Received non-JSON response")
                        response_data = {'raw_response': response.text}
                    
                    return APIResponse(
                        success=True,
                        data=response_data,
                        status_code=response.status_code
                    )
                else:
                    error_msg = f"HTTP {response.status_code}"
                    self.logger.error(f"API request failed: {error_msg}")
                    return APIResponse(
                        success=False,
                        error=error_msg,
                        status_code=response.status_code
                    )
                    
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    self.logger.warning(f"Request attempt {attempt + 1} failed. Retrying...")
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    self.logger.error(f"All retry attempts failed for {method} {endpoint}")
                    return APIResponse(
                        success=False,
                        error="Max retries exceeded"
                    )
        
        return APIResponse(success=False, error="Max retries exceeded")

    async def _make_request_async(self, method: str, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """Make async HTTP request with retry logic"""
        if not self.enable_async:
            raise RuntimeError("Async support not enabled")
        
        url = urljoin(self.base_url, endpoint)
        session = await self.api_client.get_async_session()
        
        for attempt in range(self.max_retries + 1):
            try:
                if method.upper() == 'GET':
                    async with session.get(url) as response:
                        response_data = await self._parse_async_response(response)
                elif method.upper() == 'POST':
                    async with session.post(url, json=data) as response:
                        response_data = await self._parse_async_response(response)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                if response.status < 400:
                    return APIResponse(
                        success=True,
                        data=response_data,
                        status_code=response.status
                    )
                else:
                    error_msg = f"HTTP {response.status}: {response_data}"
                    return APIResponse(
                        success=False,
                        error=error_msg,
                        status_code=response.status
                    )
                    
            except aiohttp.ClientError as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return APIResponse(success=False, error=str(e))
        
        return APIResponse(success=False, error="Max retries exceeded")
    
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
            session_id=session_id,
            agent_id=agent_id,
            start_time=start_time.isoformat(),
            user_id=user_id,
            metadata=metadata
        ))
        
        response = self._make_request('POST', '/conversations/start', data)
        
        if response.success:
            # Cache session info for validation
            with self._cache_lock:
                self._session_cache[session_id] = SessionInfo(
                    agent_id=agent_id,
                    start_time=start_time,
                    user_id=user_id
                )
            
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
            session_id=session_id,
            agent_id=agent_id,
            start_time=start_time.isoformat(),
            user_id=user_id,
            metadata=metadata
        ))
        
        response = await self._make_request_async('POST', '/conversations/start', data)
        
        if response.success:
            # Cache session info for validation
            with self._cache_lock:
                self._session_cache[session_id] = SessionInfo(
                    agent_id=agent_id,
                    start_time=start_time,
                    user_id=user_id
                )
            
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
        """End a conversation with session validation and sliding TTL"""
        # Check if session exists in cache
        session_info = None
        agent_id = None
        start_time = None
        
        with self._cache_lock:
            if session_id in self._session_cache:
                session_info = self._session_cache[session_id]
                
                # Update last access time (sliding TTL)
                session_info.touch()
                self.logger.debug(f"Session {session_id} accessed - TTL timer reset")
                
                agent_id = session_info.agent_id
                start_time = session_info.start_time
                # Remove from cache as conversation is ending
                del self._session_cache[session_id]
            else:
                self.logger.warning(f"Session {session_id} not found in cache. Proceeding to send to backend anyway.")
                # Try to extract agent_id from session_id format: agent_id_timestamp_random
                try:
                    parts = session_id.split('_')
                    if len(parts) >= 2:
                        agent_id = parts[0]
                        timestamp = int(parts[1])
                        start_time = datetime.fromtimestamp(timestamp)
                    else:
                        # Fallback: ask user to provide agent_id
                        self.logger.error(f"Cannot extract agent_id from session_id format: {session_id}")
                        return False
                except (ValueError, IndexError) as e:
                    self.logger.error(f"Cannot parse session_id {session_id}: {e}")
                    return False
        
        end_time = datetime.now()
        
        # Calculate duration
        if start_time:
            duration_seconds = (end_time - start_time).total_seconds()
            start_time_iso = start_time.isoformat()
        else:
            duration_seconds = 0.0
            start_time_iso = end_time.isoformat()
        
        # Handle quality score conversion
        quality_value = None
        if quality_score is not None:
            if isinstance(quality_score, int):
                quality_enum = ConversationQuality.from_int_safe(quality_score)
                quality_value = quality_enum.value if quality_enum else None
            elif isinstance(quality_score, ConversationQuality):
                quality_value = quality_score.value
        
        data = asdict(ConversationEndData(
            session_id=session_id,
            agent_id=agent_id,
            start_time=start_time_iso,
            end_time=end_time.isoformat(),
            duration_seconds=duration_seconds,
            status="completed",
            quality_score=quality_value,
            user_feedback=user_feedback,
            message_count=message_count,
            metadata=metadata
        ))
        
        response = self._make_request('POST', '/conversations/end', data)
        
        if response.success:
            self.logger.info(f"Conversation {session_id} completed successfully")
            return True
        else:
            self.logger.error(f"Failed to end conversation {session_id}: {response.error}")
            return False
    
    def record_failed_session(self, session_id: str, error_message: str,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record a failed conversation session with sliding TTL support"""
        # Check if session exists in cache
        session_info = None
        agent_id = None
        start_time = None
        
        with self._cache_lock:
            if session_id in self._session_cache:
                session_info = self._session_cache[session_id]
                
                # Update last access time (sliding TTL)
                session_info.touch()
                self.logger.debug(f"Failed session {session_id} accessed - TTL timer reset")
                
                agent_id = session_info.agent_id
                start_time = session_info.start_time
                # Remove from cache as conversation is ending
                del self._session_cache[session_id]
            else:
                self.logger.warning(f"Failed session {session_id} not found in cache. Proceeding to send to backend anyway.")
                # Try to extract agent_id from session_id format
                try:
                    parts = session_id.split('_')
                    if len(parts) >= 2:
                        agent_id = parts[0]
                        timestamp = int(parts[1])
                        start_time = datetime.fromtimestamp(timestamp)
                    else:
                        self.logger.error(f"Cannot extract agent_id from session_id format: {session_id}")
                        return False
                except (ValueError, IndexError) as e:
                    self.logger.error(f"Cannot parse session_id {session_id}: {e}")
                    return False
        
        end_time = datetime.now()
        
        # Calculate duration
        if start_time:
            duration_seconds = (end_time - start_time).total_seconds()
            start_time_iso = start_time.isoformat()
        else:
            duration_seconds = 0.0
            start_time_iso = end_time.isoformat()
        
        data = asdict(ConversationEndData(
            session_id=session_id,
            agent_id=agent_id,
            start_time=start_time_iso,
            end_time=end_time.isoformat(),
            duration_seconds=duration_seconds,
            status="failed",
            error_message=error_message,
            metadata=metadata
        ))
        
        response = self._make_request('POST', '/conversations/end', data)
        
        if response.success:
            self.logger.info(f"Failed conversation {session_id} recorded")
            return True
        else:
            self.logger.error(f"Failed to record failed session {session_id}: {response.error}")
            return False
    
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
    
    async def close_async(self):
        """Close resources asynchronously"""
        # Stop cleanup timer
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        # Clear session cache
        with self._cache_lock:
            self._session_cache.clear()
        
        if hasattr(self, 'api_client') and self.api_client._async_session:
            await self.api_client._async_session.close()
            self.api_client._async_session = None
        self.logger.info("AgentPerformanceTracker closed securely")

    def close(self):
        """Close resources securely"""
        # Stop cleanup timer
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        # Clear session cache
        with self._cache_lock:
            self._session_cache.clear()
        
        if hasattr(self, 'api_client'):
            self.api_client.close()
        self.logger.info("AgentPerformanceTracker closed securely")

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
