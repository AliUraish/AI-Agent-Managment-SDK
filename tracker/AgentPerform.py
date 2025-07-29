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
    """
    
    def __init__(self, 
                 base_url: str,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_async: bool = False,
                 logger: Optional[logging.Logger] = None):
        """Initialize the Agent Performance Tracker"""
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
                "AgentPerformanceTracker initialized with base URL: %s",
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
            self.logger.info(f"Conversation {session_id} started for agent {agent_id}")
            return session_id
        else:
            self.logger.error(f"Failed to start conversation: {response.error}")
            return None
    
    def end_conversation(self, session_id: str, agent_id: str,
                        start_time: Optional[datetime] = None,
                        quality_score: Optional[Union[int, ConversationQuality]] = None,
                        user_feedback: Optional[str] = None,
                        message_count: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """End a conversation and send metrics to backend"""
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
            self.logger.info(f"Conversation {session_id} completed successfully")
            return True
        else:
            self.logger.error(f"Failed to end conversation {session_id}: {response.error}")
            return False
    
    def record_failed_session(self, session_id: str, agent_id: str, error_message: str,
                             start_time: Optional[datetime] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record a failed conversation session"""
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
        if hasattr(self, 'api_client') and self.api_client._async_session:
            await self.api_client._async_session.close()
            self.api_client._async_session = None
        self.logger.info("AgentPerformanceTracker closed securely")

    def close(self):
        """Close resources securely"""
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
