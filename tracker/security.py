#!/usr/bin/env python3
"""
Security Add-on Module for AI Agent Tracking SDK

This module provides security features as a pure add-on without modifying
the core SDK. It implements:

1. Tamper detection through file checksum verification
2. Unclosed session metrics for security monitoring  
3. Security flags injection into session events
4. Security event queueing and backend communication
5. OpenTelemetry integration for observability (Phase 1)
6. PII detection hooks (Phase 2 preparation)

Usage:
    # Wrap existing tracker with security
    base_tracker = AgentPerformanceTracker(...)
    secure_tracker = SecurityWrapper(base_tracker, enable_security=True)
    
    # Use normally - security features work automatically
    session_id = secure_tracker.start_conversation("agent1", "user1")
"""

import os
import hashlib
import logging
import threading
import asyncio
import time
import uuid
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
import requests
import aiohttp

# OpenTelemetry imports (with graceful fallback)
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.trace import SpanAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Create mock classes for when OTEL is not available
    class MockTracer:
        def start_span(self, name, **kwargs):
            return MockSpan()
    
    class MockSpan:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def set_attribute(self, key, value): pass
        def set_status(self, status): pass
        def record_exception(self, exception): pass
    
    class MockStatus:
        def __init__(self, status_code, description=""): pass
    
    class MockStatusCode:
        ERROR = "ERROR"
        OK = "OK"
    
    # Mock the Status and StatusCode for when OTEL is not available
    Status = MockStatus
    StatusCode = MockStatusCode

@dataclass
class SecurityFlags:
    """Security flags for session events"""
    tamper_detected: bool = False
    pii_detected: bool = False  # For PII detection (Phase 2)
    compliance_violation: bool = False  # For future implementation
    
    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)

# ============ PII DETECTION (Phase 2 Implementation) ============

def luhn_checksum(card_number: str) -> bool:
    """
    Validate credit card number using Luhn algorithm
    
    Args:
        card_number: Credit card number string (digits only)
        
    Returns:
        bool: True if valid according to Luhn algorithm
    """
    def luhn_digit(n):
        return n if n < 10 else n - 9
    
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    
    checksum = sum(digits[-1::-2] + [luhn_digit(d * 2) for d in digits[-2::-2]])
    return checksum % 10 == 0

def detect_pii(text: str, enable_advanced: bool = True) -> bool:
    """
    Detect Personally Identifiable Information in text using regex patterns
    
    Phase 2 Implementation: Detects common PII patterns:
    - Email addresses
    - Phone numbers (US/International formats)
    - Credit card numbers (with Luhn validation)
    - Social Security Numbers
    - IP addresses
    
    Args:
        text: Text to scan for PII
        enable_advanced: Whether to use advanced detection (for performance tuning)
        
    Returns:
        bool: True if PII is detected, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Basic patterns (always checked)
    basic_patterns = [
        # Email addresses
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        
        # Phone numbers (more specific US and international formats)
        r'\b(?:\+?1[-.\s]?)?\(?[2-9][0-9]{2}\)?[-.\s]?[2-9][0-9]{2}[-.\s]?[0-9]{4}\b',
        r'\b(?:\+[1-9]\d{0,3}[-.\s]?)?(?:\([0-9]{3,4}\)[-.\s]?)?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b',
        
        # Social Security Numbers (must be 9 digits in XXX-XX-XXXX or XXXXXXXXX format)
        r'\b\d{3}-\d{2}-\d{4}\b',
        r'\b(?<!\d)\d{9}(?!\d)\b',
        
        # IP addresses (IPv4 - more strict validation)
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
    ]
    
    # Check basic patterns
    for pattern in basic_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Advanced patterns (more computationally expensive)
    if enable_advanced:
        # Credit card detection with Luhn validation
        credit_card_patterns = [
            r'\b4[0-9]{12}(?:[0-9]{3})?\b',  # Visa
            r'\b5[1-5][0-9]{14}\b',          # MasterCard
            r'\b3[47][0-9]{13}\b',           # American Express
            r'\b3[0-9]{13}\b',               # Diners Club
            r'\b6(?:011|5[0-9]{2})[0-9]{12}\b', # Discover
        ]
        
        for pattern in credit_card_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                card_number = match.group().replace(' ', '').replace('-', '')
                if luhn_checksum(card_number):
                    return True
        
        # Additional advanced patterns
        advanced_patterns = [
            # Driver's License (US format examples)
            r'\b[A-Z]{1,2}[0-9]{6,8}\b',
            
            # Passport numbers (US format)
            r'\b[0-9]{9}\b',
            
            # Bank account numbers (US routing + account)
            r'\b[0-9]{9}\s+[0-9]{8,17}\b',
        ]
        
        for pattern in advanced_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
    
    return False

def scan_metadata_for_pii(metadata: Optional[Dict[str, Any]], enable_advanced: bool = True) -> bool:
    """
    Scan metadata dictionary for PII content with performance optimization
    
    Args:
        metadata: Dictionary to scan for PII
        enable_advanced: Whether to use advanced detection patterns
        
    Returns:
        bool: True if PII is detected in any metadata value
    """
    if not metadata:
        return False
    
    def _scan_value(value, depth=0):
        # Prevent infinite recursion
        if depth > 10:
            return False
            
        if isinstance(value, str):
            return detect_pii(value, enable_advanced)
        elif isinstance(value, dict):
            # Recursively scan nested dictionaries
            for v in value.values():
                if _scan_value(v, depth + 1):
                    return True
        elif isinstance(value, list):
            # Scan list items
            for item in value[:50]:  # Limit to first 50 items for performance
                if _scan_value(item, depth + 1):
                    return True
        
        return False
    
    return _scan_value(metadata)

@dataclass
class UnclosedSessionInfo:
    """Information about unclosed sessions for security metrics"""
    session_id: str
    start_time: str
    agent_id: str
    duration_hours: float

@dataclass
class SecurityMetricEvent:
    """Security metric event data structure"""
    event_type: str
    timestamp: str
    agent_id: str
    client_id: str
    unclosed_count: int
    unclosed_sessions: List[Dict[str, Any]]

@dataclass
class TamperDetectionEvent:
    """Tamper detection event data structure"""
    event_type: str
    timestamp: str
    agent_id: str
    client_id: str
    sdk_version: str
    checksum_expected: str
    checksum_actual: str
    modified_files: List[str]

@dataclass
class SecurityAPIResponse:
    """Security API response structure"""
    success: bool
    status_code: int = 0
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SecurityManager:
    """Manages security features for the AI Agent Tracking SDK"""
    
    def __init__(self, client_id: str, sdk_version: str = "1.2.1", 
                 verbose_security_logs: bool = False,
                 enable_advanced_pii: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Security Manager
        
        Args:
            client_id: Unique identifier for this SDK client instance
            sdk_version: Version of the SDK
            verbose_security_logs: Enable verbose logging for security events
            enable_advanced_pii: Enable advanced PII detection (performance impact)
            logger: Optional logger instance
        """
        self.client_id = client_id
        self.sdk_version = sdk_version
        self.verbose_security_logs = verbose_security_logs
        self.enable_advanced_pii = enable_advanced_pii
        self.logger = logger or logging.getLogger(__name__)
        
        # Adjust logging level based on verbose flag
        if not verbose_security_logs:
            # Reduce debug/info logging in production
            security_logger = logging.getLogger(f"{__name__}.{client_id}")
            security_logger.setLevel(logging.WARNING)
            self.logger = security_logger
        
        # Security state
        self._tamper_detected = False
        self._pii_detected = False
        self._security_lock = threading.RLock()
        
        # Expected checksums for SDK files
        self._expected_checksums: Dict[str, str] = {}
        self._sdk_files: List[str] = []
        
        # Initialize tamper detection
        self._discover_sdk_files()
        self._generate_expected_checksums()
        self._perform_initial_tamper_check()
        
        if self.verbose_security_logs:
            self.logger.info(f"SecurityManager initialized for {client_id} with verbose logging")
    
    def _discover_sdk_files(self):
        """Discover main SDK Python files for checksum verification"""
        try:
            # Get the directory where this security.py file is located
            sdk_dir = Path(__file__).parent
            
            # List of main SDK files to monitor
            main_files = [
                'AgentOper.py',
                'AgentPerform.py', 
                'security.py',
                '__init__.py'
            ]
            
            for filename in main_files:
                file_path = sdk_dir / filename
                if file_path.exists():
                    self._sdk_files.append(str(file_path))
                    if self.verbose_security_logs:
                        self.logger.debug(f"Added SDK file for monitoring: {filename}")
            
            if self.verbose_security_logs:
                self.logger.info(f"Discovered {len(self._sdk_files)} SDK files for tamper detection")
            
        except Exception as e:
            self.logger.error(f"Error discovering SDK files: {e}")
    
    def _generate_expected_checksums(self):
        """Generate expected SHA256 checksums for SDK files"""
        try:
            for file_path in self._sdk_files:
                checksum = self._calculate_file_checksum(file_path)
                if checksum:
                    filename = Path(file_path).name
                    self._expected_checksums[filename] = checksum
                    if self.verbose_security_logs:
                        self.logger.debug(f"Generated checksum for {filename}: {checksum[:16]}...")
            
            if self.verbose_security_logs:
                self.logger.info(f"Generated checksums for {len(self._expected_checksums)} files")
            
        except Exception as e:
            self.logger.error(f"Error generating expected checksums: {e}")
    
    def _calculate_file_checksum(self, file_path: str) -> Optional[str]:
        """Calculate SHA256 checksum for a file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {file_path}: {e}")
            return None
    
    def _perform_initial_tamper_check(self):
        """Perform initial tamper detection during initialization"""
        modified_files = self._check_tampering()
        if modified_files:
            with self._security_lock:
                self._tamper_detected = True
            self.logger.critical(f"TAMPER DETECTION: {len(modified_files)} files modified: {modified_files}")
        else:
            if self.verbose_security_logs:
                self.logger.info("Tamper detection: All files verified successfully")
    
    def _check_tampering(self) -> List[str]:
        """Check for tampering and return list of modified files"""
        try:
            modified_files = []
            
            for file_path in self._sdk_files:
                filename = Path(file_path).name
                expected_checksum = self._expected_checksums.get(filename)
                
                if not expected_checksum:
                    continue
                
                current_checksum = self._calculate_file_checksum(file_path)
                
                if current_checksum != expected_checksum:
                    modified_files.append(filename)
                    self.logger.warning(f"Tamper detected in {filename}")
            
            return modified_files
                
        except Exception as e:
            self.logger.error(f"Error during tamper detection: {e}")
            return []
    
    def scan_for_pii(self, text: Optional[str] = None, 
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Scan text and/or metadata for PII content with performance optimization
        
        Args:
            text: Optional text to scan
            metadata: Optional metadata dict to scan
            
        Returns:
            bool: True if PII is detected
        """
        pii_found = False
        
        # Scan text if provided
        if text and detect_pii(text, self.enable_advanced_pii):
            pii_found = True
            if self.verbose_security_logs:
                self.logger.warning(f"PII detected in text content")
        
        # Scan metadata if provided
        if metadata and scan_metadata_for_pii(metadata, self.enable_advanced_pii):
            pii_found = True
            if self.verbose_security_logs:
                self.logger.warning(f"PII detected in metadata")
        
        # Update PII detection state
        if pii_found:
            with self._security_lock:
                self._pii_detected = True
                self.logger.warning("PII detected - security flags updated")
        
        return pii_found
    
    def create_tamper_detection_event(self, agent_id: str) -> TamperDetectionEvent:
        """Create a tamper detection event"""
        modified_files = self._check_tampering()
        
        # Get the first modified file for checksum comparison
        first_modified = modified_files[0] if modified_files else "unknown"
        expected_checksum = self._expected_checksums.get(first_modified, "unknown")
        
        # Calculate current checksum for the first modified file
        actual_checksum = "unknown"
        if modified_files:
            for file_path in self._sdk_files:
                if Path(file_path).name == first_modified:
                    actual_checksum = self._calculate_file_checksum(file_path) or "unknown"
                    break
        
        return TamperDetectionEvent(
            event_type="tamper_detected",
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            client_id=self.client_id,
            sdk_version=self.sdk_version,
            checksum_expected=expected_checksum,
            checksum_actual=actual_checksum,
            modified_files=modified_files
        )
    
    def create_unclosed_sessions_metric(self, agent_id: str, 
                                       unclosed_sessions: List[Dict[str, Any]]) -> SecurityMetricEvent:
        """Create an unclosed sessions security metric event"""
        return SecurityMetricEvent(
            event_type="unclosed_sessions",
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            client_id=self.client_id,
            unclosed_count=len(unclosed_sessions),
            unclosed_sessions=unclosed_sessions
        )
    
    def get_security_flags(self) -> SecurityFlags:
        """Get current security flags for session events"""
        with self._security_lock:
            return SecurityFlags(
                tamper_detected=self._tamper_detected,
                pii_detected=self._pii_detected,
                compliance_violation=False  # Future implementation
            )
    
    def is_tamper_detected(self) -> bool:
        """Check if tamper has been detected"""
        with self._security_lock:
            return self._tamper_detected
    
    def is_pii_detected(self) -> bool:
        """Check if PII has been detected"""
        with self._security_lock:
            return self._pii_detected
    
    def reset_pii_detection(self):
        """Reset PII detection state (useful for testing or session boundaries)"""
        with self._security_lock:
            self._pii_detected = False
            self.logger.debug("PII detection state reset")
    
    def recheck_tampering(self) -> List[str]:
        """Manually recheck for tampering and return list of modified files"""
        modified_files = self._check_tampering()
        if modified_files:
            with self._security_lock:
                self._tamper_detected = True
        return modified_files
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security status summary"""
        with self._security_lock:
            return {
                "client_id": self.client_id,
                "sdk_version": self.sdk_version,
                "tamper_detected": self._tamper_detected,
                "pii_detected": self._pii_detected,
                "monitored_files": len(self._sdk_files),
                "files_with_checksums": len(self._expected_checksums),
                "last_check_time": datetime.now().isoformat(),
                "security_flags": self.get_security_flags().to_dict()
            }

class SecurityAPIClient:
    """Handles security-specific API communications"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize Security API Client"""
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self._session: Optional[requests.Session] = None
        self._async_session: Optional[aiohttp.ClientSession] = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with API key"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> SecurityAPIResponse:
        """Make HTTP request to security endpoint"""
        try:
            if not self._session:
                self._session = requests.Session()
            
            url = f"{self.base_url}{endpoint}"
            headers = self._get_headers()
            
            response = self._session.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            return SecurityAPIResponse(
                success=response.status_code == 200,
                status_code=response.status_code,
                data=response.json() if response.content else None,
                error=None if response.status_code == 200 else f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            self.logger.error(f"Security API request failed: {e}")
            return SecurityAPIResponse(
                success=False,
                status_code=0,
                error=str(e)
            )
    
    async def _make_request_async(self, method: str, endpoint: str, 
                                data: Optional[Dict] = None) -> SecurityAPIResponse:
        """Make async HTTP request to security endpoint"""
        try:
            if not self._async_session:
                self._async_session = aiohttp.ClientSession()
            
            url = f"{self.base_url}{endpoint}"
            headers = self._get_headers()
            
            async with self._async_session.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                response_data = None
                if response.content_length and response.content_length > 0:
                    response_data = await response.json()
                
                return SecurityAPIResponse(
                    success=response.status == 200,
                    status_code=response.status,
                    data=response_data,
                    error=None if response.status == 200 else f"HTTP {response.status}"
                )
                
        except Exception as e:
            self.logger.error(f"Security API async request failed: {e}")
            return SecurityAPIResponse(
                success=False,
                status_code=0,
                error=str(e)
            )
    
    def send_tamper_detection(self, event: TamperDetectionEvent) -> SecurityAPIResponse:
        """Send tamper detection event to backend"""
        return self._make_request("POST", "/security/tamper", asdict(event))
    
    def send_unclosed_sessions_metric(self, event: SecurityMetricEvent) -> SecurityAPIResponse:
        """Send unclosed sessions metric to backend"""
        return self._make_request("POST", "/security/metrics", asdict(event))
    
    async def send_tamper_detection_async(self, event: TamperDetectionEvent) -> SecurityAPIResponse:
        """Send tamper detection event to backend (async)"""
        return await self._make_request_async("POST", "/security/tamper", asdict(event))
    
    async def send_unclosed_sessions_metric_async(self, event: SecurityMetricEvent) -> SecurityAPIResponse:
        """Send unclosed sessions metric to backend (async)"""
        return await self._make_request_async("POST", "/security/metrics", asdict(event))
    
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

class SecurityWrapper:
    """
    Security wrapper that adds security features to existing trackers
    
    This wrapper intercepts method calls to add security functionality
    without modifying the original tracker classes.
    """
    
    def __init__(self, tracker, enable_security: bool = True, 
                 client_id: Optional[str] = None,
                 security_check_interval: int = 300,  # 5 minutes
                 enable_tracing: bool = True,
                 otlp_endpoint: Optional[str] = None,
                 verbose_security_logs: bool = False,
                 enable_advanced_pii: bool = True,
                 batch_unclosed_sessions: bool = True,
                 max_unclosed_batch_size: int = 100,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Security Wrapper
        
        Args:
            tracker: The base tracker to wrap (AgentPerformanceTracker or AgentOperationsTracker)
            enable_security: Whether to enable security features
            client_id: Unique client identifier for security tracking
            security_check_interval: Interval for periodic security checks (seconds)
            enable_tracing: Whether to enable OpenTelemetry tracing
            otlp_endpoint: OTLP collector endpoint for traces
            verbose_security_logs: Enable verbose logging for security events
            enable_advanced_pii: Enable advanced PII detection (performance impact)
            batch_unclosed_sessions: Whether to batch unclosed session reports
            max_unclosed_batch_size: Maximum number of unclosed sessions to report at once
            logger: Optional logger instance
        """
        self.tracker = tracker
        self.enable_security = enable_security
        self.enable_tracing = enable_tracing
        self.verbose_security_logs = verbose_security_logs
        self.enable_advanced_pii = enable_advanced_pii
        self.batch_unclosed_sessions = batch_unclosed_sessions
        self.max_unclosed_batch_size = max_unclosed_batch_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Generate client ID if not provided
        self.client_id = client_id or f"security_client_{uuid.uuid4().hex[:12]}"
        
        # Initialize tracing
        if self.enable_tracing:
            self.tracer = OTELTracer(
                service_name="ai-agent-sdk",
                otlp_endpoint=otlp_endpoint,
                enable_tracing=True
            )
            if self.verbose_security_logs:
                self.logger.info(f"OTEL tracing enabled for client: {self.client_id}")
        else:
            self.tracer = OTELTracer(enable_tracing=False)
        
        # Initialize security components if enabled
        if self.enable_security:
            try:
                # Get base URL and API key from wrapped tracker
                base_url = getattr(tracker, 'base_url', 'https://api.example.com')
                api_key = getattr(tracker.api_client, '_api_key', None) if hasattr(tracker, 'api_client') else None
                
                self.security_manager = SecurityManager(
                    client_id=self.client_id,
                    sdk_version="1.2.1",
                    verbose_security_logs=self.verbose_security_logs,
                    enable_advanced_pii=self.enable_advanced_pii,
                    logger=self.logger
                )
                
                self.security_api = SecurityAPIClient(
                    base_url=base_url,
                    api_key=api_key,
                    logger=self.logger
                )
                
                # Track unclosed sessions
                self._unclosed_sessions: Dict[str, Dict[str, Any]] = {}
                self._sessions_lock = threading.RLock()
                
                # Offline event queue
                self._offline_queue = deque()
                self._queue_lock = threading.RLock()
                self._backend_available = True
                
                # Batched unclosed sessions tracking
                self._unclosed_sessions_batch: List[Dict[str, Any]] = []
                self._batch_lock = threading.RLock()
                
                # Start periodic security checks
                self._start_security_daemon(security_check_interval)
                
                # Send initial tamper detection if needed
                if self.security_manager.is_tamper_detected():
                    self._handle_tamper_detection("system_init")
                
                if self.verbose_security_logs:
                    self.logger.info(f"Security wrapper enabled for client: {self.client_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize security features: {e}")
                self.security_manager = None
                self.security_api = None
                self.enable_security = False
        else:
            self.security_manager = None
            self.security_api = None
            if self.verbose_security_logs:
                self.logger.info("Security wrapper disabled")
        
        # Initialize daemon control
        self._daemon_stop_event = threading.Event()
        self._daemon_thread: Optional[threading.Thread] = None
    
    def _start_security_daemon(self, interval: int):
        """Start the security monitoring daemon"""
        if not self.enable_security:
            return
            
        self._daemon_thread = threading.Thread(
            target=self._security_daemon,
            args=(interval,),
            daemon=True
        )
        self._daemon_thread.start()
        self.logger.info("Security monitoring daemon started")
    
    def _security_daemon(self, interval: int):
        """Security monitoring daemon that runs periodic checks"""
        while not self._daemon_stop_event.is_set():
            try:
                # Check for tampering
                if self.security_manager:
                    modified_files = self.security_manager.recheck_tampering()
                    if modified_files:
                        self._handle_tamper_detection("periodic_check")
                
                # Send unclosed sessions metric
                self._send_unclosed_sessions_metric()
                
                # Process offline queue
                self._process_offline_queue()
                
                # Wait for next cycle
                if self._daemon_stop_event.wait(timeout=interval):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in security daemon: {e}")
                if self._daemon_stop_event.wait(timeout=60):  # Wait 1 minute on error
                    break
    
    def _handle_tamper_detection(self, agent_id: str):
        """Handle tamper detection by sending alert to backend"""
        if not self.security_manager or not self.security_api:
            return
        
        # Start OTEL span for security operation
        with self.tracer.trace_security_operation(
            operation="tamper_detection",
            event_type="tamper_detected",
            client_id=self.client_id,
            agent_id=agent_id
        ) as span:
            try:
                event = self.security_manager.create_tamper_detection_event(agent_id)
                
                response = self.security_api.send_tamper_detection(event)
                if response.success:
                    if span:
                        span.set_attribute("operation.success", True)
                        span.set_attribute("modified_files_count", len(event.modified_files))
                    self.logger.critical(f"SECURITY ALERT: Tamper detection sent for {len(event.modified_files)} files")
                else:
                    if span:
                        span.set_attribute("operation.success", False)
                        span.set_attribute("error", response.error or "Unknown error")
                    self.logger.error(f"Failed to send tamper detection: {response.error}")
                    self._queue_offline_event('tamper_detected', asdict(event))
            
            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                self.logger.error(f"Error handling tamper detection: {e}")
    
    def _send_unclosed_sessions_metric(self):
        """Send unclosed sessions security metric with batching support"""
        if not self.security_manager or not self.security_api:
            return
        
        # Start OTEL span for security operation
        with self.tracer.trace_security_operation(
            operation="unclosed_sessions_metric",
            event_type="unclosed_sessions",
            client_id=self.client_id
        ) as span:
            try:
                unclosed_list = []
                
                with self._sessions_lock:
                    if not self._unclosed_sessions:
                        return  # No unclosed sessions to report
                    
                    # Collect unclosed sessions data
                    for session_id, session_data in self._unclosed_sessions.items():
                        start_time = datetime.fromisoformat(session_data['start_time'])
                        duration_hours = (datetime.now() - start_time).total_seconds() / 3600
                        
                        unclosed_list.append({
                            "session_id": session_id,
                            "start_time": session_data['start_time'],
                            "agent_id": session_data['agent_id'],
                            "duration_hours": round(duration_hours, 2)
                        })
                
                if not unclosed_list:
                    return
                
                if self.batch_unclosed_sessions and len(unclosed_list) > self.max_unclosed_batch_size:
                    # Send in batches for performance
                    batches = [unclosed_list[i:i + self.max_unclosed_batch_size] 
                              for i in range(0, len(unclosed_list), self.max_unclosed_batch_size)]
                    
                    total_sent = 0
                    for batch in batches:
                        event = self.security_manager.create_unclosed_sessions_metric(
                            agent_id="security_daemon",
                            unclosed_sessions=batch
                        )
                        
                        response = self.security_api.send_unclosed_sessions_metric(event)
                        if response.success:
                            total_sent += len(batch)
                            if self.verbose_security_logs:
                                self.logger.info(f"Sent unclosed sessions batch: {len(batch)} sessions")
                        else:
                            if span:
                                span.set_attribute("operation.success", False)
                                span.set_attribute("error", response.error or "Unknown error")
                            self.logger.error(f"Failed to send unclosed sessions batch: {response.error}")
                            self._queue_offline_event('unclosed_sessions', asdict(event))
                            break
                    
                    if span and total_sent > 0:
                        span.set_attribute("operation.success", True)
                        span.set_attribute("unclosed_sessions_count", total_sent)
                        span.set_attribute("batches_sent", len(batches))
                    
                    if total_sent > 0:
                        self.logger.warning(f"SECURITY METRIC: {total_sent} unclosed sessions reported in {len(batches)} batches")
                
                else:
                    # Send all at once (small dataset)
                    event = self.security_manager.create_unclosed_sessions_metric(
                        agent_id="security_daemon",
                        unclosed_sessions=unclosed_list
                    )
                    
                    response = self.security_api.send_unclosed_sessions_metric(event)
                    if response.success:
                        if span:
                            span.set_attribute("operation.success", True)
                            span.set_attribute("unclosed_sessions_count", len(unclosed_list))
                        self.logger.warning(f"SECURITY METRIC: {len(unclosed_list)} unclosed sessions reported")
                    else:
                        if span:
                            span.set_attribute("operation.success", False)
                            span.set_attribute("error", response.error or "Unknown error")
                        self.logger.error(f"Failed to send unclosed sessions metric: {response.error}")
                        self._queue_offline_event('unclosed_sessions', asdict(event))
            
            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                self.logger.error(f"Error sending unclosed sessions metric: {e}")
    
    def _queue_offline_event(self, event_type: str, event_data: Dict[str, Any]):
        """Queue event for later transmission when backend is available"""
        with self._queue_lock:
            self._offline_queue.append({
                'type': event_type,
                'data': event_data,
                'timestamp': datetime.now().isoformat(),
                'retry_count': 0
            })
            self._backend_available = False
            self.logger.info(f"Queued {event_type} event for offline transmission")
    
    def _process_offline_queue(self):
        """Process queued events when backend becomes available"""
        if not self._offline_queue:
            return
        
        with self._queue_lock:
            while self._offline_queue:
                event = self._offline_queue.popleft()
                
                try:
                    if event['type'] == 'tamper_detected':
                        response = self.security_api.send_tamper_detection(
                            TamperDetectionEvent(**event['data'])
                        )
                    elif event['type'] == 'unclosed_sessions':
                        response = self.security_api.send_unclosed_sessions_metric(
                            SecurityMetricEvent(**event['data'])
                        )
                    else:
                        continue
                    
                    if response.success:
                        self.logger.info(f"Replayed {event['type']} event successfully")
                        self._backend_available = True
                    else:
                        # Requeue with retry limit
                        event['retry_count'] += 1
                        if event['retry_count'] < 3:
                            self._offline_queue.appendleft(event)
                        else:
                            self.logger.error(f"Dropping {event['type']} event after 3 retries")
                        break
                
                except Exception as e:
                    self.logger.error(f"Error processing offline event: {e}")
                    break
    
    def _inject_security_flags(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Inject security flags into data payload"""
        if self.enable_security and self.security_manager:
            security_flags = self.security_manager.get_security_flags()
            data["security_flags"] = security_flags.to_dict()
        return data
    
    def _track_session_start(self, session_id: str, agent_id: str):
        """Track session start for unclosed sessions monitoring"""
        if not self.enable_security:
            return
        
        with self._sessions_lock:
            self._unclosed_sessions[session_id] = {
                'agent_id': agent_id,
                'start_time': datetime.now().isoformat(),
                'status': 'active'
            }
    
    def _track_session_end(self, session_id: str):
        """Track session end - remove from unclosed sessions"""
        if not self.enable_security:
            return
        
        with self._sessions_lock:
            if session_id in self._unclosed_sessions:
                del self._unclosed_sessions[session_id]
    
    # ============ WRAPPER METHODS FOR AgentPerformanceTracker ============
    
    def start_conversation(self, agent_id: str, user_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start conversation with security features and OTEL tracing"""
        # Start OTEL span
        with self.tracer.trace_session_operation(
            operation="start_conversation",
            agent_id=agent_id,
            client_id=self.client_id,
            user_id=user_id
        ) as span:
            try:
                # PII detection on metadata
                if self.enable_security and self.security_manager:
                    self.security_manager.scan_for_pii(metadata=metadata)
                
                # Call original method with correct signature
                session_id = self.tracker.start_conversation(agent_id, user_id, metadata)
                
                # Add session_id to span
                if session_id and span:
                    span.set_attribute("session.id", session_id)
                    span.set_attribute("operation.success", True)
                
                # Add security tracking
                if session_id and self.enable_security:
                    self._track_session_start(session_id, agent_id)
                
                return session_id
                
            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                self.logger.error(f"Error in start_conversation: {e}")
                raise
    
    def end_conversation(self, session_id: str, quality_score = None,
                        user_feedback: Optional[str] = None,
                        message_count: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """End conversation with security features and OTEL tracing"""
        # Extract agent_id from session_id for tracing
        agent_id = session_id.split('_')[0] if session_id and '_' in session_id else "unknown"
        
        # Start OTEL span
        with self.tracer.trace_session_operation(
            operation="end_conversation",
            agent_id=agent_id,
            session_id=session_id,
            client_id=self.client_id,
            quality_score=str(quality_score) if quality_score else None,
            message_count=message_count
        ) as span:
            try:
                # PII detection on user feedback and metadata
                if self.enable_security and self.security_manager:
                    self.security_manager.scan_for_pii(text=user_feedback, metadata=metadata)
                
                # Call original method
                result = self.tracker.end_conversation(session_id, quality_score, user_feedback, message_count, metadata)
                
                # Update span
                if span:
                    span.set_attribute("operation.success", result)
                
                # Add security tracking
                if result and self.enable_security:
                    self._track_session_end(session_id)
                
                return result
                
            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                self.logger.error(f"Error in end_conversation: {e}")
                raise
    
    def record_failed_session(self, session_id: str, error_message: str,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record failed session with security features and OTEL tracing"""
        # Extract agent_id from session_id for tracing
        agent_id = session_id.split('_')[0] if session_id and '_' in session_id else "unknown"
        
        # Start OTEL span
        with self.tracer.trace_session_operation(
            operation="record_failed_session",
            agent_id=agent_id,
            session_id=session_id,
            client_id=self.client_id,
            error_type="session_failure",
            error_message=error_message[:100]  # Truncate for span attribute
        ) as span:
            try:
                # PII detection on error message and metadata
                if self.enable_security and self.security_manager:
                    self.security_manager.scan_for_pii(text=error_message, metadata=metadata)
                
                # Call original method
                result = self.tracker.record_failed_session(session_id, error_message, metadata)
                
                # Update span
                if span:
                    span.set_attribute("operation.success", result)
                
                # Add security tracking
                if result and self.enable_security:
                    self._track_session_end(session_id)
                
                return result
                
            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                self.logger.error(f"Error in record_failed_session: {e}")
                raise
    
    # ============ ASYNC WRAPPER METHODS ============
    
    async def start_conversation_async(self, agent_id: str, user_id: Optional[str] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start conversation with security features (async)"""
        # Start OTEL span
        with self.tracer.trace_session_operation(
            operation="start_conversation_async",
            agent_id=agent_id,
            client_id=self.client_id,
            user_id=user_id
        ) as span:
            try:
                # PII detection on metadata
                if self.enable_security and self.security_manager:
                    self.security_manager.scan_for_pii(metadata=metadata)
                
                session_id = await self.tracker.start_conversation_async(agent_id, user_id, metadata)
                
                # Add session_id to span
                if session_id and span:
                    span.set_attribute("session.id", session_id)
                    span.set_attribute("operation.success", True)
                
                if session_id and self.enable_security:
                    self._track_session_start(session_id, agent_id)
                
                return session_id
                
            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                self.logger.error(f"Error in start_conversation_async: {e}")
                raise
    
    async def end_conversation_async(self, session_id: str, quality_score = None,
                                   user_feedback: Optional[str] = None,
                                   message_count: Optional[int] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """End conversation with security features (async)"""
        # Extract agent_id from session_id for tracing
        agent_id = session_id.split('_')[0] if session_id and '_' in session_id else "unknown"
        
        # Start OTEL span
        with self.tracer.trace_session_operation(
            operation="end_conversation_async",
            agent_id=agent_id,
            session_id=session_id,
            client_id=self.client_id,
            quality_score=str(quality_score) if quality_score else None,
            message_count=message_count
        ) as span:
            try:
                # PII detection on user feedback and metadata
                if self.enable_security and self.security_manager:
                    self.security_manager.scan_for_pii(text=user_feedback, metadata=metadata)
                
                result = await self.tracker.end_conversation_async(session_id, quality_score, user_feedback, message_count, metadata)
                
                # Update span
                if span:
                    span.set_attribute("operation.success", result)
                
                if result and self.enable_security:
                    self._track_session_end(session_id)
                
                return result
                
            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                self.logger.error(f"Error in end_conversation_async: {e}")
                raise
    
    # ============ PASSTHROUGH METHODS ============
    
    def __getattr__(self, name):
        """Pass through any other method calls to the wrapped tracker"""
        # Avoid infinite recursion by checking if attribute exists in wrapper first
        if name.startswith('_daemon_') or name in ['_sessions_lock', '_offline_queue', '_queue_lock']:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        return getattr(self.tracker, name)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics"""
        if not self.enable_security:
            return {
                "security_enabled": False,
                "tracing_enabled": self.enable_tracing,
                "otel_available": OTEL_AVAILABLE
            }
        
        with self._sessions_lock:
            unclosed_count = len(self._unclosed_sessions)
        
        with self._queue_lock:
            queue_size = len(self._offline_queue)
        
        base_stats = {
            "security_enabled": True,
            "tracing_enabled": self.enable_tracing,
            "otel_available": OTEL_AVAILABLE,
            "client_id": self.client_id,
            "unclosed_sessions_count": unclosed_count,
            "offline_queue_size": queue_size,
            "backend_available": self._backend_available,
            "daemon_running": self._daemon_thread.is_alive() if self._daemon_thread else False
        }
        
        if self.security_manager:
            base_stats.update(self.security_manager.get_security_summary())
        
        return base_stats
    
    def close(self):
        """Close security wrapper and underlying tracker"""
        # Stop security daemon
        if self._daemon_thread and self._daemon_thread.is_alive():
            self._daemon_stop_event.set()
            self._daemon_thread.join(timeout=5)
        
        # Close security API client
        if self.security_api:
            self.security_api.close()
        
        # Close underlying tracker
        if hasattr(self.tracker, 'close'):
            self.tracker.close()
        
        self.logger.info("Security wrapper closed")
    
    async def close_async(self):
        """Close security wrapper and underlying tracker (async)"""
        # Stop security daemon
        if self._daemon_thread and self._daemon_thread.is_alive():
            self._daemon_stop_event.set()
            self._daemon_thread.join(timeout=5)
        
        # Close security API client
        if self.security_api:
            await self.security_api.close_async()
        
        # Close underlying tracker
        if hasattr(self.tracker, 'close_async'):
            await self.tracker.close_async()
        elif hasattr(self.tracker, 'close'):
            self.tracker.close()
        
        self.logger.info("Security wrapper closed (async)")

# ============ FACTORY FUNCTIONS ============

def create_secure_performance_tracker(base_url: str, api_key: Optional[str] = None,
                                     enable_security: bool = True,
                                     enable_tracing: bool = True,
                                     otlp_endpoint: Optional[str] = None,
                                     verbose_security_logs: bool = False,
                                     enable_advanced_pii: bool = True,
                                     batch_unclosed_sessions: bool = True,
                                     max_unclosed_batch_size: int = 100,
                                     client_id: Optional[str] = None,
                                     **tracker_kwargs) -> SecurityWrapper:
    """
    Factory function to create a secure AgentPerformanceTracker
    
    Args:
        base_url: API base URL
        api_key: API authentication key
        enable_security: Whether to enable security features
        enable_tracing: Whether to enable OpenTelemetry tracing
        otlp_endpoint: OTLP collector endpoint for traces
        verbose_security_logs: Enable verbose logging for security events
        enable_advanced_pii: Enable advanced PII detection (performance impact)
        batch_unclosed_sessions: Whether to batch unclosed session reports
        max_unclosed_batch_size: Maximum number of unclosed sessions to report at once
        client_id: Unique client identifier for security tracking
        **tracker_kwargs: Additional arguments for AgentPerformanceTracker
    
    Returns:
        SecurityWrapper wrapping AgentPerformanceTracker
    """
    from .AgentPerform import AgentPerformanceTracker
    
    base_tracker = AgentPerformanceTracker(base_url, api_key, **tracker_kwargs)
    return SecurityWrapper(
        tracker=base_tracker, 
        enable_security=enable_security,
        enable_tracing=enable_tracing,
        otlp_endpoint=otlp_endpoint,
        verbose_security_logs=verbose_security_logs,
        enable_advanced_pii=enable_advanced_pii,
        batch_unclosed_sessions=batch_unclosed_sessions,
        max_unclosed_batch_size=max_unclosed_batch_size,
        client_id=client_id
    )

def create_secure_operations_tracker(base_url: str, api_key: Optional[str] = None,
                                    enable_security: bool = True,
                                    enable_tracing: bool = True,
                                    otlp_endpoint: Optional[str] = None,
                                    verbose_security_logs: bool = False,
                                    enable_advanced_pii: bool = True,
                                    batch_unclosed_sessions: bool = True,
                                    max_unclosed_batch_size: int = 100,
                                    client_id: Optional[str] = None,
                                    **tracker_kwargs) -> SecurityWrapper:
    """
    Factory function to create a secure AgentOperationsTracker
    
    Args:
        base_url: API base URL
        api_key: API authentication key
        enable_security: Whether to enable security features
        enable_tracing: Whether to enable OpenTelemetry tracing
        otlp_endpoint: OTLP collector endpoint for traces
        verbose_security_logs: Enable verbose logging for security events
        enable_advanced_pii: Enable advanced PII detection (performance impact)
        batch_unclosed_sessions: Whether to batch unclosed session reports
        max_unclosed_batch_size: Maximum number of unclosed sessions to report at once
        client_id: Unique client identifier for security tracking
        **tracker_kwargs: Additional arguments for AgentOperationsTracker
    
    Returns:
        SecurityWrapper wrapping AgentOperationsTracker
    """
    from .AgentOper import AgentOperationsTracker
    
    base_tracker = AgentOperationsTracker(base_url, api_key, **tracker_kwargs)
    return SecurityWrapper(
        tracker=base_tracker,
        enable_security=enable_security,
        enable_tracing=enable_tracing,
        otlp_endpoint=otlp_endpoint,
        verbose_security_logs=verbose_security_logs,
        enable_advanced_pii=enable_advanced_pii,
        batch_unclosed_sessions=batch_unclosed_sessions,
        max_unclosed_batch_size=max_unclosed_batch_size,
        client_id=client_id
    )

# ============ OPENTELEMETRY INTEGRATION ============

class OTELTracer:
    """OpenTelemetry tracer wrapper for SDK operations"""
    
    def __init__(self, service_name: str = "ai-agent-sdk", 
                 otlp_endpoint: Optional[str] = None,
                 enable_tracing: bool = True):
        """
        Initialize OTEL tracer
        
        Args:
            service_name: Service name for tracing
            otlp_endpoint: OTLP collector endpoint (optional)
            enable_tracing: Whether to enable tracing
        """
        self.enabled = enable_tracing and OTEL_AVAILABLE
        self.service_name = service_name
        
        if self.enabled:
            try:
                # Set up resource
                resource = Resource.create({
                    "service.name": service_name,
                    "service.version": "1.2.1"
                })
                
                # Configure tracer provider
                trace.set_tracer_provider(TracerProvider(resource=resource))
                
                # Set up exporter if endpoint provided
                if otlp_endpoint:
                    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
                    span_processor = BatchSpanProcessor(otlp_exporter)
                    trace.get_tracer_provider().add_span_processor(span_processor)
                
                self.tracer = trace.get_tracer(__name__)
                
            except Exception as e:
                # Fallback to mock tracer if setup fails
                self.enabled = False
                self.tracer = MockTracer()
        else:
            self.tracer = MockTracer()
    
    def start_span(self, name: str, **attributes) -> Any:
        """Start a new tracing span"""
        if self.enabled:
            span = self.tracer.start_span(name)
            
            # Set common attributes
            span.set_attribute("service.name", self.service_name)
            
            # Set custom attributes
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, str(value))
            
            return span
        else:
            return MockSpan()
    
    def trace_session_operation(self, operation: str, agent_id: str, 
                              session_id: Optional[str] = None,
                              client_id: Optional[str] = None,
                              **extra_attributes):
        """Create a span for session operations"""
        attributes = {
            "operation.type": "session",
            "operation.name": operation,
            "agent.id": agent_id,
            "session.id": session_id,
            "client.id": client_id,
        }
        attributes.update(extra_attributes)
        
        return self.start_span(f"sdk.session.{operation}", **attributes)
    
    def trace_security_operation(self, operation: str, event_type: str,
                               client_id: Optional[str] = None,
                               **extra_attributes):
        """Create a span for security operations"""
        attributes = {
            "operation.type": "security", 
            "operation.name": operation,
            "security.event_type": event_type,
            "client.id": client_id,
        }
        attributes.update(extra_attributes)
        
        return self.start_span(f"sdk.security.{operation}", **attributes)
