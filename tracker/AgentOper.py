import time
import uuid
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum, auto
import requests
import aiohttp
from urllib.parse import urljoin
import re


@dataclass
class APIResponse:
    """Standard response structure from backend API"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


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

@dataclass
class APIConfig:
    """API configuration with defaults"""
    base_url: str = "http://localhost:8080/api"
    api_key: str = "sdk_api_key_2024_permanent_3301f4d913aa2e2a928a3686bdc17e33"
    client_id: str = "sdk_client_permanent_2024"
    max_retries: int = 3
    retry_delay: float = 0.5  # Initial delay in seconds
    max_delay: float = 8.0    # Maximum delay in seconds
    timeout: float = 30.0     # Request timeout in seconds
    rate_limit: int = 5000    # Requests per minute

class SecureAPIClient:
    """Secure API client with rate limiting and validation"""
    
    def __init__(self, base_url: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 client_id: Optional[str] = None,
                 config: Optional[APIConfig] = None):
        """Initialize API client with configuration"""
        self.config = config or APIConfig()
        
        # Override config with explicit parameters if provided
        if base_url:
            self.config.base_url = base_url.rstrip('/')
        if api_key:
            self.config.api_key = api_key
        if client_id:
            self.config.client_id = client_id
            
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=self.config.rate_limit,
            time_window=60  # 1 minute window
        )
        
        # Initialize session objects
        self._session: Optional[requests.Session] = None
        self._async_session: Optional[aiohttp.ClientSession] = None
        self._session_lock = threading.Lock()
        self._async_session_lock = asyncio.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.secure_logger = SecureLogger()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key"""
        return {
            'Authorization': f'Bearer {self.config.api_key}',
            'X-Client-ID': self.config.client_id,
            'Content-Type': 'application/json',
            'User-Agent': f'SDK-Agent-Tracker/1.0 (Client: {self.config.client_id})'
        }
    
    def get_session(self) -> requests.Session:
        """Get or create requests session"""
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    self._session = requests.Session()
                    self._session.headers.update(self._get_headers())
        return self._session
    
    async def get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async session"""
        if self._async_session is None:
            async with self._async_session_lock:
                if self._async_session is None:
                    self._async_session = aiohttp.ClientSession(
                        headers=self._get_headers()
                    )
        return self._async_session
    
    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """Make HTTP request with rate limiting and retries"""
        # Validate metadata if present
        if data and 'metadata' in data:
            try:
                data['metadata'] = validate_metadata(data['metadata'])
            except ValidationError as e:
                return APIResponse(success=False, error=str(e))
        
        # Convert any datetime objects to ms timestamps
        if data:
            for key, value in data.items():
                if isinstance(value, datetime):
                    data[key] = to_ms_timestamp(value)
        
        # Wait for rate limit
        self.rate_limiter.wait_if_needed()
        
        url = urljoin(self.config.base_url, endpoint.lstrip('/'))
        retry_count = 0
        delay = self.config.retry_delay
        
        while retry_count <= self.config.max_retries:
            try:
                session = self.get_session()
                response = session.request(
                    method=method,
                    url=url,
                    json=data if data else None,
                    timeout=self.config.timeout
                )
                
                # Log request (securely)
                self.secure_logger.debug(
                    f"API Request: {method} {endpoint} - Status: {response.status_code}",
                    extra={'data': {'endpoint': endpoint, 'status': response.status_code}}
                )
                
                if response.status_code == 429:  # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', delay))
                    time.sleep(retry_after)
                    continue
                
                try:
                    response_data = response.json() if response.content else None
                except ValueError:
                    response_data = None
                
                if response.ok:
                    return APIResponse(
                        success=True,
                        status_code=response.status_code,
                        data=response_data
                    )
                else:
                    error_msg = response_data.get('error') if response_data else response.text
                    return APIResponse(
                        success=False,
                        status_code=response.status_code,
                        error=error_msg
                    )
                    
            except (requests.Timeout, requests.ConnectionError) as e:
                self.logger.warning(f"Request failed (attempt {retry_count + 1}/{self.config.max_retries}): {e}")
                if retry_count == self.config.max_retries:
                    return APIResponse(success=False, error=str(e))
                    
                time.sleep(delay)
                delay = min(delay * 2, self.config.max_delay)  # Exponential backoff
                retry_count += 1
                continue
                
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                return APIResponse(success=False, error=str(e))
    
    async def make_request_async(self, method: str, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """Make async HTTP request with rate limiting and retries"""
        # Validate metadata if present
        if data and 'metadata' in data:
            try:
                data['metadata'] = validate_metadata(data['metadata'])
            except ValidationError as e:
                return APIResponse(success=False, error=str(e))
        
        # Convert any datetime objects to ms timestamps
        if data:
            for key, value in data.items():
                if isinstance(value, datetime):
                    data[key] = to_ms_timestamp(value)
        
        # Wait for rate limit
        self.rate_limiter.wait_if_needed()
        
        url = urljoin(self.config.base_url, endpoint.lstrip('/'))
        retry_count = 0
        delay = self.config.retry_delay
        
        while retry_count <= self.config.max_retries:
            try:
                session = await self.get_async_session()
                async with session.request(
                    method=method,
                    url=url,
                    json=data if data else None,
                    timeout=self.config.timeout
                ) as response:
                    
                    # Log request (securely)
                    self.secure_logger.debug(
                        f"API Request (async): {method} {endpoint} - Status: {response.status}",
                        extra={'data': {'endpoint': endpoint, 'status': response.status}}
                    )
                    
                    if response.status == 429:  # Rate limit exceeded
                        retry_after = int(response.headers.get('Retry-After', delay))
                        await asyncio.sleep(retry_after)
                        continue
                    
                    try:
                        response_data = await response.json() if await response.content.read() else None
                    except ValueError:
                        response_data = None
                    
                    if response.ok:
                        return APIResponse(
                            success=True,
                            status_code=response.status,
                            data=response_data
                        )
                    else:
                        error_msg = response_data.get('error') if response_data else await response.text()
                        return APIResponse(
                            success=False,
                            status_code=response.status,
                            error=error_msg
                        )
                    
            except (aiohttp.ClientTimeout, aiohttp.ClientConnectionError) as e:
                self.logger.warning(f"Async request failed (attempt {retry_count + 1}/{self.config.max_retries}): {e}")
                if retry_count == self.config.max_retries:
                    return APIResponse(success=False, error=str(e))
                    
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.max_delay)  # Exponential backoff
                retry_count += 1
                continue
                
            except Exception as e:
                self.logger.error(f"Unexpected error in async request: {e}")
                return APIResponse(success=False, error=str(e))
    
    def close(self):
        """Close synchronous session"""
        if self._session:
            self._session.close()
            self._session = None
    
    async def close_async(self):
        """Close asynchronous session"""
        if self._async_session:
            await self._async_session.close()
            self._async_session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_async()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class AgentStatus(Enum):
    """Valid agent status values"""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    
    @classmethod
    def from_str(cls, value: str) -> 'AgentStatus':
        """Convert string to AgentStatus, with validation"""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid agent status: {value}. Must be one of: {', '.join([s.value for s in cls])}")
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a status string is valid"""
        return value.lower() in [s.value for s in cls]

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_quality_score(score: float) -> float:
    """Validate quality score (1.0 to 5.0)"""
    try:
        score_float = float(score)
        if not 1.0 <= score_float <= 5.0:
            raise ValidationError(f"Quality score must be between 1.0 and 5.0, got: {score}")
        return round(score_float, 2)
    except (TypeError, ValueError):
        raise ValidationError(f"Quality score must be a number, got: {score}")

def validate_response_time(time_ms: float) -> float:
    """Validate response time (milliseconds, non-negative)"""
    try:
        time_float = float(time_ms)
        if time_float < 0:
            raise ValidationError(f"Response time cannot be negative, got: {time_ms}")
        return round(time_float, 2)
    except (TypeError, ValueError):
        raise ValidationError(f"Response time must be a number, got: {time_ms}")

def to_ms_timestamp(dt: Optional[datetime] = None) -> int:
    """Convert datetime to millisecond timestamp"""
    if dt is None:
        dt = datetime.now()
    return int(dt.timestamp() * 1000)

def validate_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ensure metadata is JSON-serializable"""
    if metadata is None:
        return None
    try:
        # Test JSON serialization
        json.dumps(metadata)
        return metadata
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Metadata must be JSON-serializable: {e}")

class RateLimiter:
    """Rate limiter implementation"""
    def __init__(self, max_requests: int = 5000, time_window: int = 60):
        self.max_requests = max_requests  # requests per window
        self.time_window = time_window    # window in seconds
        self.requests = []
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Try to acquire a rate limit token"""
        with self.lock:
            now = time.time()
            # Remove old requests
            self.requests = [req for req in self.requests if now - req < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                return False
                
            self.requests.append(now)
            return True
    
    def wait_if_needed(self):
        """Wait until rate limit allows request"""
        while not self.acquire():
            time.sleep(0.1)  # Wait 100ms before retry


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
class ActivityLogData:
    """Data structure for activity logging"""
    agent_id: str
    action: str
    timestamp: str
    details: Dict[str, Any]
    duration: Optional[float] = None


class AgentOperationsTracker:
    """
    Agent Operations Tracker - Handles only operational aspects:
    - Active agents monitoring
    - Agent status management  
    - Recent activity logging
    """
    
    def __init__(self, 
                 base_url: str,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_async: bool = False,
                 logger: Optional[logging.Logger] = None):
        """Initialize the Agent Operations Tracker"""
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
    
    def update_agent_status(self, agent_id: str, status: Union[AgentStatus, str], 
                           previous_status: Optional[Union[AgentStatus, str]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update agent status in the backend"""
        status_value = status.value if isinstance(status, AgentStatus) else status
        previous_status_value = previous_status.value if isinstance(previous_status, AgentStatus) else previous_status
        
        data = asdict(AgentStatusData(
            agent_id=agent_id,
            status=status_value,
            timestamp=datetime.now().isoformat(),
            previous_status=previous_status_value,
            metadata=metadata
        ))
        
        response = self._make_request('POST', '/agents/status', data)
        
        if response.success:
            self.logger.debug(f"Agent {agent_id} status updated to {status_value}")
            return True
        else:
            self.logger.error(f"Failed to update agent {agent_id} status: {response.error}")
            return False
    
    async def update_agent_status_async(self, agent_id: str, status: Union[AgentStatus, str],
                                       previous_status: Optional[Union[AgentStatus, str]] = None,
                                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async version of update_agent_status"""
        status_value = status.value if isinstance(status, AgentStatus) else status
        previous_status_value = previous_status.value if isinstance(previous_status, AgentStatus) else previous_status
        
        data = asdict(AgentStatusData(
            agent_id=agent_id,
            status=status_value,
            timestamp=datetime.now().isoformat(),
            previous_status=previous_status_value,
            metadata=metadata
        ))
        
        response = await self._make_request_async('POST', '/agents/status', data)
        
        if response.success:
            self.logger.debug(f"Agent {agent_id} status updated to {status_value}")
            return True
        else:
            self.logger.error(f"Failed to update agent {agent_id} status: {response.error}")
            return False
    
    def log_activity(self, agent_id: str, activity_type: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Log agent activity to the backend"""
        data = asdict(ActivityLogData(
            agent_id=agent_id,
            action=activity_type,
            timestamp=datetime.now().isoformat(),
            details=metadata or {},
            duration=None
        ))
        
        response = self._make_request('POST', '/agents/activity', data)
        
        if response.success:
            self.logger.debug(f"Activity logged for agent {agent_id}: {activity_type}")
            return True
        else:
            self.logger.error(f"Failed to log activity for agent {agent_id}: {response.error}")
            return False
    
    async def log_activity_async(self, agent_id: str, activity_type: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async version of log_activity"""
        data = asdict(ActivityLogData(
            agent_id=agent_id,
            action=activity_type,
            timestamp=datetime.now().isoformat(),
            details=metadata or {},
            duration=None
        ))
        
        response = await self._make_request_async('POST', '/agents/activity', data)
        
        if response.success:
            self.logger.debug(f"Activity logged for agent {agent_id}: {activity_type}")
            return True
        else:
            self.logger.error(f"Failed to log activity for agent {agent_id}: {response.error}")
            return False
    
    def get_active_agents(self) -> Optional[Dict[str, Any]]:
        """Get list of currently active agents"""
        response = self._make_request('GET', '/agents/active')
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get active agents: {response.error}")
            return None
    
    async def get_active_agents_async(self) -> Optional[Dict[str, Any]]:
        """Async version of get_active_agents"""
        response = await self._make_request_async('GET', '/agents/active')
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get active agents: {response.error}")
            return None
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a specific agent"""
        response = self._make_request('GET', f'/agents/{agent_id}/status')
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get agent status for {agent_id}: {response.error}")
            return None
    
    async def get_agent_status_async(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Async version of get_agent_status"""
        response = await self._make_request_async('GET', f'/agents/{agent_id}/status')
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get agent status for {agent_id}: {response.error}")
            return None
    
    def get_recent_activity(self, limit: int = 50, agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get recent activity logs"""
        endpoint = '/agents/activity'
        if agent_id:
            endpoint = f'/agents/{agent_id}/activity'
        
        params = {'limit': limit}
        response = self._make_request('GET', f"{endpoint}?limit={limit}")
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get recent activity: {response.error}")
            return None
    
    async def get_recent_activity_async(self, limit: int = 50, agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Async version of get_recent_activity"""
        endpoint = '/agents/activity'
        if agent_id:
            endpoint = f'/agents/{agent_id}/activity'
        
        response = await self._make_request_async('GET', f"{endpoint}?limit={limit}")
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get recent activity: {response.error}")
            return None
    
    def get_operations_overview(self) -> Optional[Dict[str, Any]]:
        """Get operations overview (active agents, status distribution, recent activity summary)"""
        response = self._make_request('GET', '/operations/overview')
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get operations overview: {response.error}")
            return None
    
    async def get_operations_overview_async(self) -> Optional[Dict[str, Any]]:
        """Async version of get_operations_overview"""
        response = await self._make_request_async('GET', '/operations/overview')
        
        if response.success:
            return response.data
        else:
            self.logger.error(f"Failed to get operations overview: {response.error}")
            return None
    
    async def close_async(self):
        """Close resources asynchronously"""
        if hasattr(self, 'api_client') and self.api_client._async_session:
            await self.api_client._async_session.close()
            self.api_client._async_session = None
        self.logger.info("AgentOperationsTracker closed securely")

    def close(self):
        """Close resources securely"""
        if hasattr(self, 'api_client'):
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
        await self.close_async()
