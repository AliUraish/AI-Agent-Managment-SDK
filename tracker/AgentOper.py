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

    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """Public method for making HTTP requests"""
        return self._make_request(method, endpoint, data)
    
    async def make_request_async(self, method: str, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """Public method for making async HTTP requests"""
        return await self._make_request_async(method, endpoint, data)
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """Make HTTP request with secure handling"""
        url = urljoin(self.base_url, endpoint)
        
        # Mask sensitive data in logs
        secure_logger = SecureLogger()
        log_data = secure_logger.mask_sensitive_data(data) if data else None
        
        for attempt in range(3):  # Default 3 retries
            try:
                session = self.get_session()
                
                if method.upper() == 'GET':
                    response = session.get(url, timeout=30)
                elif method.upper() == 'POST':
                    response = session.post(url, json=data, timeout=30)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Parse response
                if response.status_code < 400:
                    try:
                        response_data = response.json() if response.content else {}
                    except json.JSONDecodeError:
                        response_data = {'raw_response': response.text}
                    
                    return APIResponse(
                        success=True,
                        data=response_data,
                        status_code=response.status_code
                    )
                else:
                    error_msg = f"HTTP {response.status_code}"
                    return APIResponse(
                        success=False,
                        error=error_msg,
                        status_code=response.status_code
                    )
                    
            except requests.exceptions.RequestException as e:
                if attempt < 2:  # Retry on failure
                    time.sleep(1.0 * (2 ** attempt))
                else:
                    return APIResponse(
                        success=False,
                        error=str(e)
                    )
        
        return APIResponse(success=False, error="Max retries exceeded")

    async def _make_request_async(self, method: str, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """Make async HTTP request with retry logic"""
        url = urljoin(self.base_url, endpoint)
        session = await self.get_async_session()
        
        for attempt in range(3):  # Default 3 retries
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
                if attempt < 2:
                    await asyncio.sleep(1.0 * (2 ** attempt))
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


class AgentStatus(Enum):
    """Enumeration for different agent statuses"""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


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
