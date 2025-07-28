import time
import uuid
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import aiohttp
from urllib.parse import urljoin
import re

class SecureLogger:
    """Secure logger that masks sensitive information"""
    
    SENSITIVE_FIELDS = ['api_key', 'token', 'secret', 'password', 'authorization']
    
    @staticmethod
    def mask_sensitive_data(data: Any) -> Any:
        """Mask sensitive information in logs"""
        if isinstance(data, dict):
            return {k: SecureLogger.mask_value(k, v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [SecureLogger.mask_sensitive_data(item) for item in data]
        elif isinstance(data, str):
            # Mask anything that looks like a bearer token or API key
            if re.match(r'^(Bearer\s+|key-|sk-|pk-|api-|token-).+', data, re.IGNORECASE):
                return f"{data[:8]}...{data[-4:]}" if len(data) > 12 else "***masked***"
        return data
    
    @staticmethod
    def mask_value(key: str, value: Any) -> Any:
        """Mask value if key is sensitive"""
        if any(sensitive in key.lower() for sensitive in SecureLogger.SENSITIVE_FIELDS):
            if isinstance(value, str):
                return f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***masked***"
            return "***masked***"
        return SecureLogger.mask_sensitive_data(value)
    
    @staticmethod
    def format_log(message: str, *args, **kwargs) -> str:
        """Format log message with masked sensitive data"""
        try:
            if args:
                args = tuple(SecureLogger.mask_sensitive_data(arg) for arg in args)
            if kwargs:
                kwargs = {k: SecureLogger.mask_sensitive_data(v) for k, v in kwargs.items()}
            return message % (*args, *kwargs) if args or kwargs else message
        except Exception:
            return message

class SecureAPIClient:
    """Secure API client that handles sensitive information"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self._api_key = api_key  # Store securely, never log
        self._session: Optional[requests.Session] = None
        self._async_session: Optional[aiohttp.ClientSession] = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with secure handling of API key"""
        headers = {'Content-Type': 'application/json'}
        if self._api_key:
            # Add API key but ensure it's not exposed in string representation
            headers['Authorization'] = f'Bearer {self._api_key}'
        return headers
    
    def get_session(self) -> requests.Session:
        """Get or create secure session"""
        if not self._session:
            self._session = requests.Session()
            self._session.headers.update(self._get_headers())
        return self._session
    
    async def get_async_session(self) -> aiohttp.ClientSession:
        """Get or create secure async session"""
        if not self._async_session or self._async_session.closed:
            self._async_session = aiohttp.ClientSession(headers=self._get_headers())
        return self._async_session
    
    def close(self):
        """Close sessions securely"""
        if self._session:
            self._session.close()
            self._session = None
        
        if self._async_session and not self._async_session.closed:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self._async_session.close())
            except RuntimeError:
                pass
            self._async_session = None


class AgentStatus(Enum):
    """Enumeration for different agent statuses"""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


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
class APIResponse:
    """Standard response structure from backend API"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


@dataclass
class AgentRegistrationData:
    """Data structure for agent registration"""
    agent_id: str
    registration_time: str
    sdk_version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentStatusData:
    """Data structure for agent status updates"""
    agent_id: str
    status: str
    timestamp: str
    previous_status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


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


class AgentOperationsTracker:
    """
    API-based Agent Operations Tracker that sends data to backend instead of storing in memory.
    
    Integrates with backend API endpoints:
    - POST /agents/register
    - POST /agents/status
    - POST /conversations/start
    - POST /conversations/end
    - GET /system/overview
    """
    
    def __init__(self, 
                 base_url: str,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_async: bool = False,
                 logger: Optional[logging.Logger] = None):
        """Initialize the API-based tracker"""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_async = enable_async
        
        # Setup secure logging
        self.logger = logger or logging.getLogger(__name__)
        self.secure_logger = SecureLogger()
        
        # Initialize secure API client
        self.api_client = SecureAPIClient(base_url, api_key)
        
        # Track active sessions locally (minimal memory usage)
        self._active_sessions: Dict[str, str] = {}
        
        # Log initialization without exposing API key
        self.logger.info(
            self.secure_logger.format_log(
                "AgentOperationsTracker initialized with base URL: %s",
                self.base_url
            )
        )
    
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
    
    def register_agent(self, agent_id: str, sdk_version: Optional[str] = None, 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register a new agent with the backend"""
        data = asdict(AgentRegistrationData(
            agent_id=agent_id,
            registration_time=datetime.now().isoformat(),
            sdk_version=sdk_version,
            metadata=metadata
        ))
        
        response = self._make_request('POST', '/agents/register', data)
        
        if response.success:
            self.logger.info(f"Agent {agent_id} registered successfully")
            return True
        else:
            self.logger.error(f"Failed to register agent {agent_id}: {response.error}")
            return False
    
    async def register_agent_async(self, agent_id: str, sdk_version: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async version of register_agent"""
        data = asdict(AgentRegistrationData(
            agent_id=agent_id,
            registration_time=datetime.now().isoformat(),
            sdk_version=sdk_version,
            metadata=metadata
        ))
        
        response = await self._make_request_async('POST', '/agents/register', data)
        
        if response.success:
            self.logger.info(f"Agent {agent_id} registered successfully")
            return True
        else:
            self.logger.error(f"Failed to register agent {agent_id}: {response.error}")
            return False
    
    def update_agent_status(self, agent_id: str, status: AgentStatus, 
                           previous_status: Optional[AgentStatus] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update agent status in the backend"""
        data = asdict(AgentStatusData(
            agent_id=agent_id,
            status=status.value,
            timestamp=datetime.now().isoformat(),
            previous_status=previous_status.value if previous_status else None,
            metadata=metadata
        ))
        
        response = self._make_request('POST', '/agents/status', data)
        
        if response.success:
            self.logger.debug(f"Agent {agent_id} status updated to {status.value}")
            return True
        else:
            self.logger.error(f"Failed to update agent {agent_id} status: {response.error}")
            return False
    
    async def update_agent_status_async(self, agent_id: str, status: AgentStatus,
                                       previous_status: Optional[AgentStatus] = None,
                                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async version of update_agent_status"""
        data = asdict(AgentStatusData(
            agent_id=agent_id,
            status=status.value,
            timestamp=datetime.now().isoformat(),
            previous_status=previous_status.value if previous_status else None,
            metadata=metadata
        ))
        
        response = await self._make_request_async('POST', '/agents/status', data)
        
        if response.success:
            self.logger.debug(f"Agent {agent_id} status updated to {status.value}")
            return True
        else:
            self.logger.error(f"Failed to update agent {agent_id} status: {response.error}")
            return False
    
    def start_conversation(self, agent_id: str, session_id: Optional[str] = None,
                          user_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start tracking a new conversation"""
        session_id = session_id or str(uuid.uuid4())
        start_time = datetime.now()
        
        data = asdict(ConversationStartData(
            session_id=session_id,
            agent_id=agent_id,
            start_time=start_time.isoformat(),
            user_id=user_id,
            metadata=metadata
        ))
        
        response = self._make_request('POST', '/conversations/start', data)
        
        if response.success:
            # Track session locally for validation
            self._active_sessions[session_id] = agent_id
            self.logger.info(f"Conversation {session_id} started for agent {agent_id}")
            return session_id
        else:
            self.logger.error(f"Failed to start conversation: {response.error}")
            return None
    
    async def start_conversation_async(self, agent_id: str, session_id: Optional[str] = None,
                                      user_id: Optional[str] = None,
                                      metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Async version of start_conversation"""
        session_id = session_id or str(uuid.uuid4())
        start_time = datetime.now()
        
        data = asdict(ConversationStartData(
            session_id=session_id,
            agent_id=agent_id,
            start_time=start_time.isoformat(),
            user_id=user_id,
            metadata=metadata
        ))
        
        response = await self._make_request_async('POST', '/conversations/start', data)
        
        if response.success:
            self._active_sessions[session_id] = agent_id
            self.logger.info(f"Conversation {session_id} started for agent {agent_id}")
            return session_id
        else:
            self.logger.error(f"Failed to start conversation: {response.error}")
            return None
    
    def end_conversation(self, session_id: str, 
                        start_time: Optional[datetime] = None,
                        quality_score: Optional[Union[int, ConversationQuality]] = None,
                        user_feedback: Optional[str] = None,
                        message_count: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """End a conversation and send metrics to backend"""
        # Validate session exists
        agent_id = self._active_sessions.get(session_id)
        if not agent_id:
            self.logger.warning(f"Attempted to end unknown session: {session_id}")
            return False
        
        end_time = datetime.now()
        
        # Calculate duration (if start_time not provided, estimate from recent time)
        if start_time is None:
            # Estimate duration as 0 if we don't have start time
            duration_seconds = 0.0
            start_time_iso = end_time.isoformat()
        else:
            duration_seconds = (end_time - start_time).total_seconds()
            start_time_iso = start_time.isoformat()
        
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
            # Remove from active sessions
            self._active_sessions.pop(session_id, None)
            self.logger.info(f"Conversation {session_id} completed successfully")
            return True
        else:
            self.logger.error(f"Failed to end conversation {session_id}: {response.error}")
            return False
    
    def record_failed_session(self, session_id: str, error_message: str,
                             start_time: Optional[datetime] = None,
                             agent_id: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record a failed conversation session"""
        # Try to get agent_id from active sessions or use provided one
        if not agent_id:
            agent_id = self._active_sessions.get(session_id, "unknown")
        
        end_time = datetime.now()
        
        # Calculate duration
        if start_time is None:
            duration_seconds = 0.0
            start_time_iso = end_time.isoformat()
        else:
            duration_seconds = (end_time - start_time).total_seconds()
            start_time_iso = start_time.isoformat()
        
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
            # Remove from active sessions
            self._active_sessions.pop(session_id, None)
            self.logger.info(f"Failed conversation {session_id} recorded")
            return True
        else:
            self.logger.error(f"Failed to record failed session {session_id}: {response.error}")
            return False
    
    def get_system_overview(self) -> Optional[Dict[str, Any]]:
        """Get system overview from backend (used by dashboard)"""
        response = self._make_request('GET', '/system/overview')
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get system overview: {response.error}")
            return None
    
    async def get_system_overview_async(self) -> Optional[Dict[str, Any]]:
        """Async version of get_system_overview"""
        response = await self._make_request_async('GET', '/system/overview')
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get system overview: {response.error}")
            return None
    
    def get_active_sessions_count(self) -> int:
        """Get count of locally tracked active sessions"""
        return len(self._active_sessions)
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is currently active (locally tracked)"""
        return session_id in self._active_sessions
    
    def close(self):
        """Close resources securely"""
        self.api_client.close()
        self.logger.info("AgentOperationsTracker closed securely")
    
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
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()
