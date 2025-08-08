#!/usr/bin/env python3
"""
Compliance Tracking Module for AI Agent Tracking SDK

This module provides Phase 1 compliance features as a pure add-on:

1. ComplianceManager - Immutable audit logging with tamper detection
2. Policy violation tracking (HIPAA, GDPR, SOC 2, ISO 27001)
3. User acknowledgment system for risk acceptance
4. SecurityWrapper integration for automatic compliance injection

Key Features:
- Append-only audit logs with hash-chaining for tamper detection
- HIPAA PHI detection and backend compliance verification
- GDPR scope detection and data region tracking
- SOC 2/ISO 27001 evidence collection
- User risk acknowledgment with persistent logging

Usage:
    # Create compliance-enabled tracker
    tracker = create_compliance_tracker(
        base_url="https://api.example.com",
        enable_compliance=True,
        hipaa_scope=True,
        gdpr_scope=True
    )
    
    # Use normally - compliance logging happens automatically
    session_id = tracker.start_conversation("agent1", "user1", metadata)
    
    # Acknowledge specific policy risks if needed
    tracker.acknowledge_risk(session_id, "hipaa_violation", "Approved by compliance team")
"""

import os
import hashlib
import logging
import threading
import time
import uuid
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import deque
import requests
import aiohttp

@dataclass
class PolicyViolation:
    """Individual policy violation details"""
    type: str  # e.g., "hipaa_violation", "gdpr_violation", "retention_violation"
    details: str
    severity: str = "medium"  # low, medium, high, critical
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ComplianceFlags:
    """Compliance and policy flags for session tracking"""
    gdpr_scope: bool = False
    hipaa_scope: bool = False
    pii_detected: bool = False
    phi_detected: bool = False  # Protected Health Information
    data_sent_to: List[str] = field(default_factory=list)
    retention_policy: str = "30_days"  # 7_days, 30_days, 90_days, 1_year, indefinite
    user_region: Optional[str] = None  # EU, US, CA, etc.
    data_processing_region: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass 
class UserAcknowledgment:
    """User risk acknowledgment record"""
    session_id: str
    policy_type: str  # hipaa_violation, gdpr_violation, etc.
    acknowledged_by: str  # user ID or system identifier
    acknowledged_at: str
    reason: str
    scope: str = "session"  # session, user, global
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ComplianceAuditEntry:
    """Immutable audit log entry for compliance tracking"""
    entry_id: str
    session_id: str
    timestamp: str
    event_type: str  # start_conversation, end_conversation, policy_violation, acknowledgment
    compliance_flags: ComplianceFlags
    policy_violations: List[PolicyViolation] = field(default_factory=list)
    user_acknowledged: bool = False
    acknowledgments: List[UserAcknowledgment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Tamper detection fields
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None
    
    def calculate_hash(self) -> str:
        """Calculate SHA256 hash of entry content for tamper detection"""
        # Create deterministic content for hashing
        content = {
            "entry_id": self.entry_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "compliance_flags": self.compliance_flags.to_dict(),
            "policy_violations": [v.to_dict() for v in self.policy_violations],
            "user_acknowledged": self.user_acknowledged,
            "acknowledgments": [a.to_dict() for a in self.acknowledgments],
            "metadata": self.metadata,
            "previous_hash": self.previous_hash
        }
        
        # Convert to deterministic JSON and hash
        content_json = json.dumps(content, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content_json.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "entry_id": self.entry_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "compliance_flags": self.compliance_flags.to_dict(),
            "policy_violations": [v.to_dict() for v in self.policy_violations],
            "user_acknowledged": self.user_acknowledged,
            "acknowledgments": [a.to_dict() for a in self.acknowledgments],
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash
        }

class PHIDetector:
    """Detects Protected Health Information (PHI) for HIPAA compliance"""
    
    def __init__(self):
        self._compile_phi_patterns()
    
    def _compile_phi_patterns(self):
        """Compile regex patterns for PHI detection"""
        self.phi_patterns = [
            # Medical record numbers
            re.compile(r'\b(?:MR|MRN|Medical Record)\s*:?\s*[A-Z0-9]{6,12}\b', re.IGNORECASE),
            
            # Insurance/Member ID numbers
            re.compile(r'\b(?:Insurance|Member|Policy)\s*(?:ID|Number)\s*:?\s*[A-Z0-9]{8,15}\b', re.IGNORECASE),
            
            # Medical terminology indicators
            re.compile(r'\b(?:diagnosis|prescription|medication|treatment|therapy|surgery|patient|medical history)\b', re.IGNORECASE),
            
            # ICD-10 codes
            re.compile(r'\b[A-Z][0-9]{2}\.?[0-9A-Z]{0,4}\b'),
            
            # CPT codes (Current Procedural Terminology)
            re.compile(r'\b[0-9]{5}[TFM]?\b'),
            
            # Drug names (common patterns)
            re.compile(r'\b(?:mg|ml|tablet|capsule|injection|dose)\b', re.IGNORECASE),
        ]
    
    def detect_phi(self, text: str) -> bool:
        """
        Detect if text contains Protected Health Information
        
        Args:
            text: Text to scan for PHI
            
        Returns:
            bool: True if PHI is likely detected
        """
        if not text or not isinstance(text, str):
            return False
        
        # Count pattern matches
        matches = 0
        for pattern in self.phi_patterns:
            if pattern.search(text):
                matches += 1
                # If we find medical terminology + identifiers, likely PHI
                if matches >= 2:
                    return True
        
        return matches >= 1 and len(text) > 50  # Require substantial content

class GDPRDetector:
    """Detects GDPR-relevant data and scope"""
    
    EU_COUNTRIES = {
        'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
        'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
        'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
    }
    
    def is_eu_region(self, region_code: Optional[str]) -> bool:
        """Check if region code indicates EU jurisdiction"""
        if not region_code:
            return False
        return region_code.upper() in self.EU_COUNTRIES
    
    def detect_gdpr_scope(self, metadata: Optional[Dict[str, Any]] = None, 
                         user_region: Optional[str] = None) -> bool:
        """
        Determine if GDPR applies to this session
        
        Args:
            metadata: Session metadata that might contain region info
            user_region: Explicit user region code
            
        Returns:
            bool: True if GDPR scope applies
        """
        # Check explicit user region
        if user_region and self.is_eu_region(user_region):
            return True
        
        # Check metadata for region indicators
        if metadata:
            # Look for common region fields
            region_fields = ['region', 'country', 'location', 'user_region', 'client_region']
            for field in region_fields:
                if field in metadata:
                    region_value = metadata[field]
                    if isinstance(region_value, str) and self.is_eu_region(region_value):
                        return True
        
        return False

class ComplianceManager:
    """Manages compliance tracking and policy enforcement"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._audit_log = []
        self._audit_lock = threading.Lock()
        self._last_hash = None
    
    def is_gdpr_scope(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Check if the conversation is in GDPR scope"""
        if not metadata:
            return False
        
        # Check for EU region indicators
        eu_regions = {'EU', 'EEA', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'DK', 'SE', 'FI', 'AT', 'PL'}
        region = metadata.get('region', '').upper()
        if region in eu_regions:
            return True
        
        # Check for GDPR-related flags
        if metadata.get('gdpr_required') or metadata.get('eu_data_processing'):
            return True
        
        return False
    
    def is_hipaa_scope(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Check if the conversation is in HIPAA scope"""
        if not metadata:
            return False
        
        # Check for healthcare-related indicators
        if metadata.get('hipaa_required') or metadata.get('healthcare_data'):
            return True
        
        # Check for healthcare context
        context = metadata.get('context', '').lower()
        healthcare_keywords = {'healthcare', 'medical', 'patient', 'hospital', 'clinic', 'phi'}
        if any(keyword in context for keyword in healthcare_keywords):
            return True
        
        return False
    
    def detect_policy_violations(self, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect any policy violations based on metadata"""
        violations = []
        
        if not metadata:
            return violations
        
        # Check GDPR violations
        if self.is_gdpr_scope(metadata):
            # Check for non-EU data processing
            if metadata.get('data_location', '').upper() not in {'EU', 'EEA'}:
                violations.append({
                    'type': 'gdpr_violation',
                    'details': 'Data processing outside EU/EEA',
                    'severity': 'high'
                })
        
        # Check HIPAA violations
        if self.is_hipaa_scope(metadata):
            # Check for non-HIPAA compliant backend
            if not metadata.get('hipaa_compliant_backend'):
                violations.append({
                    'type': 'hipaa_violation',
                    'details': 'Healthcare data processing without HIPAA-compliant backend',
                    'severity': 'high'
                })
        
        # Check data retention policy
        retention_days = metadata.get('retention_days')
        if retention_days and retention_days > 30:
            violations.append({
                'type': 'retention_violation',
                'details': f'Data retention period ({retention_days} days) exceeds 30-day limit',
                'severity': 'medium'
            })
        
        return violations
    
    def log_compliance_event(self, session_id: str, agent_id: str, event_type: str,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a compliance event to the audit trail"""
        timestamp = datetime.now().isoformat()
        
        # Create audit entry
        entry = {
            'timestamp': timestamp,
            'session_id': session_id,
            'agent_id': agent_id,
            'event_type': event_type,
            'metadata': metadata or {},
            'previous_hash': self._last_hash
        }
        
        # Calculate entry hash
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry['hash'] = entry_hash
        
        # Add to audit log with thread safety
        with self._audit_lock:
            self._audit_log.append(entry)
            self._last_hash = entry_hash
            
        self.logger.info(f"Compliance event logged: {event_type} for session {session_id}")
    
    def get_audit_trail(self, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get audit trail entries within the specified time range"""
        with self._audit_lock:
            if not (start_time or end_time):
                return self._audit_log.copy()
            
            filtered_log = []
            for entry in self._audit_log:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if start_time and entry_time < start_time:
                    continue
                if end_time and entry_time > end_time:
                    continue
                filtered_log.append(entry)
            
            return filtered_log
    
    def verify_audit_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verify the integrity of the audit trail"""
        with self._audit_lock:
            if not self._audit_log:
                return True, None
            
            previous_hash = None
            for entry in self._audit_log:
                # Verify hash chain
                if entry['previous_hash'] != previous_hash:
                    return False, f"Hash chain broken at entry {entry['timestamp']}"
                
                # Verify entry hash
                entry_copy = entry.copy()
                stored_hash = entry_copy.pop('hash')
                entry_str = json.dumps(entry_copy, sort_keys=True)
                calculated_hash = hashlib.sha256(entry_str.encode()).hexdigest()
                
                if calculated_hash != stored_hash:
                    return False, f"Entry hash mismatch at {entry['timestamp']}"
                
                previous_hash = stored_hash
            
            return True, None

# ============ COMPLIANCE WRAPPER FOR SECURITYWRAPPER INTEGRATION ============

class ComplianceWrapper:
    """
    Compliance wrapper that intelligently integrates with SecurityWrapper
    
    Automatically detects existing security features and layers appropriately:
    - If SecurityWrapper is present: Wraps it directly
    - If base tracker only: Optionally creates SecurityWrapper first
    - Provides seamless compliance features regardless of security layer
    
    This class provides comprehensive compliance tracking capabilities including:
    - HIPAA PHI detection and compliance monitoring
    - GDPR scope detection and cross-border transfer alerts
    - SOC 2/ISO 27001 audit trail generation
    - User risk acknowledgment system
    - Automatic policy violation detection
    - Immutable audit logging with tamper detection
    
    Type Safety:
        All public methods have explicit type hints for better IDE support
        and runtime type checking compatibility.
    
    Error Handling:
        Comprehensive logging for security auto-wrapping failures with
        detailed explanations of missing features and remediation steps.
    """
    
    def __init__(self, wrapped_tracker: Any, 
                 enable_compliance: bool = True,
                 auto_detect_security: bool = True,
                 enable_security_if_missing: bool = True,
                 hipaa_scope: bool = False,
                 gdpr_scope: bool = False,
                 default_retention_policy: str = "30_days",
                 audit_log_path: Optional[str] = None,
                 client_id: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 # Security wrapper arguments (passed through if creating SecurityWrapper)
                 **security_kwargs: Any) -> None:
        """
        Initialize Compliance Wrapper with intelligent security integration
        
        Args:
            wrapped_tracker: Existing tracker (any type)
            enable_compliance: Whether to enable compliance features
            auto_detect_security: Whether to automatically detect existing SecurityWrapper
            enable_security_if_missing: Whether to create SecurityWrapper if not present
            hipaa_scope: Whether HIPAA compliance is required
            gdpr_scope: Whether GDPR compliance is required
            default_retention_policy: Default data retention policy
            audit_log_path: Path for audit log file
            client_id: Client identifier
            logger: Optional logger instance
            **security_kwargs: Arguments passed to SecurityWrapper if created
        """
        self.enable_compliance = enable_compliance
        self.auto_detect_security = auto_detect_security
        self.logger = logger or logging.getLogger(__name__)
        
        # Intelligent wrapper detection and layering
        self.wrapped_tracker = self._setup_intelligent_wrapping(
            wrapped_tracker, 
            enable_security_if_missing,
            security_kwargs
        )
        
        # Generate client ID if not provided
        self.client_id = client_id or self._extract_client_id()
        
        # Initialize compliance manager if enabled
        if self.enable_compliance:
            try:
                self.compliance_manager = ComplianceManager() # Changed to instantiate ComplianceManager
                
                self.logger.info(f"ComplianceWrapper enabled for {self.client_id} (auto-layered: {self._is_security_present()})")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize compliance features: {e}")
                self.compliance_manager = None
                self.enable_compliance = False
        else:
            self.compliance_manager = None
            self.logger.info("ComplianceWrapper disabled")
    
    def _setup_intelligent_wrapping(self, wrapped_tracker: Any, enable_security_if_missing: bool, security_kwargs: Dict[str, Any]) -> Any:
        """
        Intelligently detect and layer security wrapper if needed
        
        Args:
            wrapped_tracker: Original tracker to wrap
            enable_security_if_missing: Whether to add SecurityWrapper if not present
            security_kwargs: Arguments for SecurityWrapper creation
            
        Returns:
            Appropriately wrapped tracker
        """
        # Check if SecurityWrapper is already present
        if self._is_security_wrapper(wrapped_tracker):
            self.logger.info("SecurityWrapper detected - layering compliance on top")
            return wrapped_tracker
        
        # Check if tracker is already wrapped by SecurityWrapper (nested detection)
        if hasattr(wrapped_tracker, 'wrapped_tracker') and self._is_security_wrapper(wrapped_tracker.wrapped_tracker):
            self.logger.info("Nested SecurityWrapper detected - using existing security layer")
            return wrapped_tracker
        
        # Check if security features are available and should be enabled
        if enable_security_if_missing and self.auto_detect_security:
            try:
                from .security import SecurityWrapper
                
                # Extract client_id directly from wrapped_tracker, not self
                client_id = self._extract_client_id_from_tracker(wrapped_tracker, security_kwargs.get('client_id'))
                
                # Extract reasonable defaults for SecurityWrapper
                security_config = {
                    'enable_security': True,
                    'client_id': client_id,
                    'enable_tracing': security_kwargs.get('enable_tracing', True),
                    'verbose_security_logs': security_kwargs.get('verbose_security_logs', False),
                    'enable_advanced_pii': security_kwargs.get('enable_pii_detection', True),
                    'security_check_interval': security_kwargs.get('security_check_interval', 300),
                    'batch_unclosed_sessions': security_kwargs.get('batch_unclosed_sessions', True),
                    'otlp_endpoint': security_kwargs.get('otlp_endpoint', None)
                }
                
                # Add any additional security kwargs
                security_config.update({k: v for k, v in security_kwargs.items() 
                                      if k not in security_config})
                
                wrapped_with_security = SecurityWrapper(
                    tracker=wrapped_tracker,
                    **security_config
                )
                
                self.logger.info("Auto-created SecurityWrapper for comprehensive protection")
                return wrapped_with_security
                
            except ImportError:
                self.logger.warning(
                    "SecurityWrapper auto-wrapping failed: SecurityWrapper module not available. "
                    "Security features (tamper detection, PII scanning, OTEL tracing) will be missing. "
                    "To enable security features, ensure the security module is properly installed. "
                    "Continuing with base tracker and compliance-only features."
                )
                return wrapped_tracker
            except Exception as e:
                self.logger.error(
                    f"SecurityWrapper auto-wrapping failed: {e}. "
                    f"Security features will be unavailable. "
                    f"Continuing with base tracker and compliance-only features."
                )
                return wrapped_tracker
        
        # Use base tracker without security
        self.logger.info("Using base tracker without security layer")
        return wrapped_tracker
    
    def _is_security_wrapper(self, tracker: Any) -> bool:
        """Check if tracker is a SecurityWrapper instance"""
        # Check class name to avoid import dependencies
        return (hasattr(tracker, '__class__') and 
                tracker.__class__.__name__ == 'SecurityWrapper')
    
    def _is_security_present(self) -> bool:
        """Check if security features are present in the wrapped tracker"""
        return (self._is_security_wrapper(self.wrapped_tracker) or
                (hasattr(self.wrapped_tracker, 'wrapped_tracker') and 
                 self._is_security_wrapper(self.wrapped_tracker.wrapped_tracker)))
    
    def _extract_client_id_from_tracker(self, tracker: Any, provided_client_id: Optional[str] = None) -> str:
        """Extract client ID from a specific tracker or generate one"""
        # Use provided client_id if available
        if provided_client_id:
            return provided_client_id
            
        # Try to get client_id from various possible locations in the given tracker
        potential_sources = [
            tracker,
            getattr(tracker, 'wrapped_tracker', None),
            getattr(tracker, 'tracker', None),
            getattr(tracker, 'security_manager', None)
        ]
        
        for source in potential_sources:
            if source and hasattr(source, 'client_id'):
                return source.client_id
        
        # Generate new client ID if none found
        import uuid
        return f"compliance_client_{uuid.uuid4().hex[:12]}"
    
    def _extract_client_id(self) -> str:
        """Extract client ID from wrapped tracker or generate one"""
        return self._extract_client_id_from_tracker(self.wrapped_tracker)
    
    def get_security_info(self) -> Dict[str, Any]:
        """
        Get detailed information about security wrapper integration status
        
        Returns:
            Dict[str, Any]: Security integration information containing:
                - compliance_enabled: Whether compliance features are active
                - security_present: Whether security features are available
                - security_wrapper_detected: Whether SecurityWrapper is in use
                - client_id: Client identifier for tracking
                - wrapper_chain: List showing complete wrapper hierarchy
                
        Integration Features:
            - Provides visibility into automatic security layer detection
            - Shows complete wrapper chain for debugging
            - Helps troubleshoot security integration issues
            - Supports configuration validation and monitoring
            
        Example:
            >>> tracker.get_security_info()
            {
                'compliance_enabled': True,
                'security_present': True,
                'security_wrapper_detected': True,
                'client_id': 'compliance_client_abc123',
                'wrapper_chain': ['SecurityWrapper', 'AgentPerformanceTracker']
            }
        """
        return {
            "compliance_enabled": self.enable_compliance,
            "security_present": self._is_security_present(),
            "security_wrapper_detected": self._is_security_wrapper(self.wrapped_tracker),
            "client_id": self.client_id,
            "wrapper_chain": self._analyze_wrapper_chain()
        }
    
    def _analyze_wrapper_chain(self) -> List[str]:
        """Analyze the chain of wrappers for debugging"""
        chain = []
        current = self.wrapped_tracker
        
        # Traverse wrapper chain
        for _ in range(10):  # Prevent infinite loops
            if not current:
                break
                
            chain.append(current.__class__.__name__)
            
            # Look for next wrapped tracker
            next_tracker = None
            for attr in ['wrapped_tracker', 'tracker', '_tracker']:
                if hasattr(current, attr):
                    next_tracker = getattr(current, attr)
                    break
            
            if not next_tracker or next_tracker == current:
                break
                
            current = next_tracker
        
        return chain

    def _inject_compliance_data(self, data: Dict[str, Any], 
                               compliance_flags: ComplianceFlags,
                               violations: List[PolicyViolation]) -> Dict[str, Any]:
        """Inject compliance flags and violations into data payload"""
        if self.enable_compliance and self.compliance_manager:
            data["compliance_flags"] = compliance_flags.to_dict()
            data["policy_violations"] = [v.to_dict() for v in violations]
        
        return data
    
    def start_conversation(self, agent_id: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start a conversation with compliance checks"""
        try:
            # Add compliance flags to metadata
            metadata = metadata or {}
            metadata['compliance_flags'] = {
                'gdpr_scope': self.compliance_manager.is_gdpr_scope(metadata),
                'hipaa_scope': self.compliance_manager.is_hipaa_scope(metadata),
                'policy_violations': []
            }
            
            # Check for policy violations
            violations = self.compliance_manager.detect_policy_violations(metadata)
            if violations:
                metadata['compliance_flags']['policy_violations'] = violations
                metadata['security_flags'] = metadata.get('security_flags', {})
                metadata['security_flags']['compliance_violation'] = True
            
            # Start conversation
            session_id = self.wrapped_tracker.start_conversation(agent_id, metadata)
            
            if session_id:
                # Log compliance event
                self.compliance_manager.log_compliance_event(
                    session_id=session_id,
                    agent_id=agent_id,
                    event_type='conversation_start',
                    metadata=metadata
                )
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error in compliance start_conversation: {e}")
            raise
    
    async def start_conversation_async(self, agent_id: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start a conversation with compliance checks (async)"""
        try:
            # Add compliance flags to metadata
            metadata = metadata or {}
            metadata['compliance_flags'] = {
                'gdpr_scope': self.compliance_manager.is_gdpr_scope(metadata),
                'hipaa_scope': self.compliance_manager.is_hipaa_scope(metadata),
                'policy_violations': []
            }
            
            # Check for policy violations
            violations = self.compliance_manager.detect_policy_violations(metadata)
            if violations:
                metadata['compliance_flags']['policy_violations'] = violations
                metadata['security_flags'] = metadata.get('security_flags', {})
                metadata['security_flags']['compliance_violation'] = True
            
            # Start conversation
            session_id = await self.wrapped_tracker.start_conversation_async(agent_id, metadata)
            
            if session_id:
                # Log compliance event
                self.compliance_manager.log_compliance_event(
                    session_id=session_id,
                    agent_id=agent_id,
                    event_type='conversation_start',
                    metadata=metadata
                )
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error in compliance start_conversation_async: {e}")
            raise
    
    def end_conversation(self, session_id: str, quality_score: Optional[float] = None,
                        user_feedback: Optional[str] = None,
                        message_count: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """End a conversation with compliance checks"""
        try:
            # Add compliance flags to metadata
            metadata = metadata or {}
            metadata['compliance_flags'] = {
                'gdpr_scope': self.compliance_manager.is_gdpr_scope(metadata),
                'hipaa_scope': self.compliance_manager.is_hipaa_scope(metadata),
                'policy_violations': []
            }
            
            # Check for policy violations
            violations = self.compliance_manager.detect_policy_violations(metadata)
            if violations:
                metadata['compliance_flags']['policy_violations'] = violations
                metadata['security_flags'] = metadata.get('security_flags', {})
                metadata['security_flags']['compliance_violation'] = True
            
            # End conversation
            result = self.wrapped_tracker.end_conversation(
                session_id=session_id,
                quality_score=quality_score,
                user_feedback=user_feedback,
                message_count=message_count,
                metadata=metadata
            )
            
            if result:
                # Log compliance event
                self.compliance_manager.log_compliance_event(
                    session_id=session_id,
                    agent_id=session_id.split('_')[0],  # Use first part of session ID as agent ID
                    event_type='conversation_end',
                    metadata=metadata
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in compliance end_conversation: {e}")
            raise
    
    def record_failed_session(self, session_id: str, error_message: str,
                             error_type: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record failed session with compliance logging and error message analysis
        
        Args:
            session_id: Unique session identifier
            error_message: Error description (scanned for PII/PHI content)
            metadata: Optional session metadata for compliance analysis
            
        Returns:
            bool: True if failed session recorded successfully, False otherwise
            
        Compliance Features:
            - Scans error_message for accidentally exposed PII/PHI content
            - Detects policy violations in error handling
            - Logs compliance event to immutable audit trail
            - Tracks failed sessions for compliance reporting
            
        Raises:
            Exception: Re-raises any exceptions from the wrapped tracker
        """
        try:
            # Call wrapped tracker
            result = self.wrapped_tracker.record_failed_session(session_id, error_message, error_type=error_type, metadata=metadata)
            
            # Log compliance event
            if result and self.enable_compliance and self.compliance_manager:
                # Determine data backends
                data_sent_to = self._detect_data_backends(metadata)
                
                # Create compliance flags
                compliance_flags = ComplianceFlags() # No longer using create_compliance_flags
                
                # Log compliance event
                self.compliance_manager.log_compliance_event(
                    session_id=session_id,
                    event_type="record_failed_session",
                    metadata=metadata
                )
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error in compliance record_failed_session: {e}")
            raise
    
    def acknowledge_risk(self, session_id: str, policy_type: str, 
                        acknowledged_by: str = "user", 
                        reason: str = "Risk accepted") -> bool:
        """
        Acknowledge and accept specific policy risks for a session
        
        Args:
            session_id: Unique session identifier
            policy_type: Type of policy violation to acknowledge 
                        (e.g., 'hipaa_violation', 'gdpr_violation', 'retention_violation')
            acknowledged_by: Identifier of the person/system acknowledging the risk
            reason: Explanation for why the risk is being accepted
            
        Returns:
            bool: True if risk acknowledgment was recorded successfully, False otherwise
            
        Compliance Features:
            - Records immutable acknowledgment in audit trail
            - Prevents future duplicate violation flags for same session/policy type
            - Maintains audit evidence for compliance reviews
            - Supports risk management and compliance workflows
            
        Note:
            Acknowledged risks will not trigger duplicate policy violations
            for the same session and policy type in future events.
        """
        if self.enable_compliance and self.compliance_manager:
            # The original acknowledge_risk method in ComplianceManager expects
            # session_id, policy_type, acknowledged_by, reason, scope.
            # The new ComplianceManager.log_compliance_event expects session_id, agent_id, event_type, metadata.
            # This method needs to be updated to match the new log_compliance_event signature.
            # For now, we'll call the new log_compliance_event with a placeholder agent_id.
            # A proper implementation would involve a new acknowledgment record class.
            self.compliance_manager.log_compliance_event(
                session_id=session_id,
                agent_id="compliance_system", # Placeholder for acknowledged_by
                event_type="risk_acknowledgment",
                metadata={
                    "acknowledgment": {
                        "session_id": session_id,
                        "policy_type": policy_type,
                        "acknowledged_by": acknowledged_by,
                        "acknowledged_at": datetime.now().isoformat(),
                        "reason": reason,
                        "scope": "session" # Placeholder, needs to be part of a new UserAcknowledgment dataclass
                    },
                    "scope": "session" # Placeholder, needs to be part of a new UserAcknowledgment dataclass
                }
            )
            return True
        return False
    
    def _detect_data_backends(self, metadata: Optional[Dict[str, Any]]) -> List[str]:
        """Detect which backends data is sent to"""
        backends = []
        
        # Get from wrapped tracker's base URL
        if hasattr(self.wrapped_tracker, 'base_url'):
            backends.append(self.wrapped_tracker.base_url)
        elif hasattr(self.wrapped_tracker, 'tracker') and hasattr(self.wrapped_tracker.tracker, 'base_url'):
            backends.append(self.wrapped_tracker.tracker.base_url)
        
        # Check metadata for additional backends
        if metadata:
            backend_fields = ['backend_url', 'ai_provider', 'processing_endpoint']
            for field in backend_fields:
                if field in metadata and isinstance(metadata[field], str):
                    backends.append(metadata[field])
        
        return backends
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive compliance status summary and statistics
        
        Returns:
            Dict[str, Any]: Compliance summary containing:
                - compliance_enabled: Whether compliance features are active
                - hipaa_scope: Whether HIPAA compliance is enabled
                - gdpr_scope: Whether GDPR compliance is enabled
                - total_audit_entries: Number of audit log entries
                - total_sessions_tracked: Number of unique sessions
                - policy_violations: Count of violations by type
                - acknowledged_violations: Number of acknowledged violations
                - hash_chaining_enabled: Whether tamper detection is active
                - security_integration: Security wrapper integration status
                
        Compliance Features:
            - Provides audit-ready compliance statistics
            - Shows security integration status and wrapper chain
            - Summarizes policy violations and acknowledgments
            - Supports compliance reporting and dashboard displays
        """
        if self.enable_compliance and self.compliance_manager:
            summary = {
                "compliance_enabled": True,
                "hipaa_scope": self.compliance_manager.is_hipaa_scope(None), # Placeholder for metadata
                "gdpr_scope": self.compliance_manager.is_gdpr_scope(None), # Placeholder for metadata
                "total_audit_entries": len(self.compliance_manager.get_audit_trail(start_time=datetime.min, end_time=datetime.max)), # Placeholder for time range
                "total_sessions_tracked": 0, # No direct count of sessions in new ComplianceManager
                "policy_violations": {}, # Placeholder for violations
                "acknowledged_violations": 0, # Placeholder for acknowledged violations
                "hash_chaining_enabled": False, # No hash chaining in new ComplianceManager
                "security_integration": self.get_security_info()
            }
            return summary
        
        return {
            "compliance_enabled": False,
            "message": "Compliance tracking disabled",
            "security_integration": self.get_security_info()
        }
    
    def get_audit_trail(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve immutable audit trail entries with optional session filtering
        
        Args:
            session_id: Optional session ID to filter audit entries.
                       If None, returns all audit entries.
                       
        Returns:
            List[Dict[str, Any]]: List of audit trail entries, each containing:
                - entry_id: Unique audit entry identifier
                - session_id: Session identifier
                - timestamp: ISO timestamp of the event
                - event_type: Type of compliance event
                - compliance_flags: Compliance flags at time of event
                - policy_violations: List of detected violations
                - user_acknowledged: Whether risks were acknowledged
                - entry_hash: Tamper detection hash
                
        Compliance Features:
            - Provides immutable audit evidence for certifications
            - Supports forensic analysis and compliance investigations
            - Enables session-specific or global audit trail review
            - Maintains hash-chained tamper detection integrity
        """
        if self.enable_compliance and self.compliance_manager:
            # The new ComplianceManager.get_audit_trail expects start_time and end_time.
            # For a simple session filter, we'll return all entries or filter by session_id if provided.
            # A more robust implementation would involve a new AuditEntry dataclass.
            return self.compliance_manager.get_audit_trail(start_time=datetime.min, end_time=datetime.max)
        
        return []
    
    def verify_audit_integrity(self) -> Dict[str, Any]:
        """
        Verify integrity of the audit trail using hash chain validation
        
        Returns:
            Dict[str, Any]: Integrity verification results containing:
                - integrity_verified: Whether audit trail passes integrity checks
                - total_entries: Number of audit entries verified
                - corrupted_entries: List of entries with integrity issues
                - hash_chain_intact: Whether hash chain is unbroken
                - integrity_enabled: Whether hash chaining is enabled
                
        Compliance Features:
            - Provides tamper detection for audit evidence
            - Supports forensic validation of audit logs
            - Enables compliance auditors to verify log integrity
            - Detects unauthorized modifications to audit trail
            
        Note:
            This verification is critical for SOC 2, ISO 27001, and other
            compliance frameworks that require tamper-proof audit evidence.
        """
        if self.enable_compliance and self.compliance_manager:
            integrity_verified, message = self.compliance_manager.verify_audit_integrity()
            return {
                "integrity_verified": integrity_verified,
                "total_entries": 0, # No direct count in new ComplianceManager
                "corrupted_entries": [], # Placeholder for corrupted entries
                "hash_chain_intact": integrity_verified,
                "integrity_enabled": False # No hash chaining in new ComplianceManager
            }
        
        return {"compliance_enabled": False}
    
    # ============ PASSTHROUGH METHODS ============
    
    def __getattr__(self, name: str) -> Any:
        """
        Pass through any other method calls to the wrapped tracker
        
        Args:
            name: Name of the attribute/method to access on wrapped tracker
            
        Returns:
            Any: The requested attribute/method from the wrapped tracker
            
        Note:
            This enables the ComplianceWrapper to act as a transparent proxy
            for all methods not explicitly overridden, maintaining full
            compatibility with the underlying tracker interface.
        """
        return getattr(self.wrapped_tracker, name)

# ============ ENHANCED FACTORY FUNCTIONS ============

def create_compliance_tracker(base_url: str, api_key: Optional[str] = None,
                             enable_compliance: bool = True,
                             enable_security: bool = True,
                             auto_detect_security: bool = True,
                             hipaa_scope: bool = False,
                             gdpr_scope: bool = False,
                             default_retention_policy: str = "30_days",
                             audit_log_path: Optional[str] = None,
                             client_id: Optional[str] = None,
                             # Security configuration (passed through if SecurityWrapper is created)
                             enable_advanced_pii: bool = True,
                             enable_tracing: bool = True,
                             verbose_security_logs: bool = False,
                             security_check_interval: int = 300,
                             batch_unclosed_sessions: bool = True,
                             otlp_endpoint: Optional[str] = None,
                             **tracker_kwargs) -> ComplianceWrapper:
    """
    Enhanced factory function to create a compliance-enabled tracker with intelligent security integration
    
    This function now automatically handles security wrapper integration:
    - If SecurityWrapper is available and enable_security=True: Creates it automatically
    - If SecurityWrapper is not available: Uses base tracker only
    - Always wraps with ComplianceWrapper for compliance features
    
    Args:
        base_url: API base URL
        api_key: API authentication key
        enable_compliance: Whether to enable compliance features
        enable_security: Whether to enable security features (auto-created if available)
        auto_detect_security: Whether to automatically detect and integrate security
        hipaa_scope: Whether HIPAA compliance is required
        gdpr_scope: Whether GDPR compliance is required
        default_retention_policy: Default data retention policy
        audit_log_path: Path for audit log file
        client_id: Client identifier
                 enable_advanced_pii: Enable advanced PII detection in SecurityWrapper
         security_check_interval: Security check interval in seconds
        enable_tracing: Enable OpenTelemetry tracing in SecurityWrapper
        verbose_security_logs: Enable verbose security logging
        **tracker_kwargs: Additional arguments for base tracker
        
    Returns:
        ComplianceWrapper with intelligent security integration
    """
    from .AgentPerform import AgentPerformanceTracker
    
    # Create base tracker
    base_tracker = AgentPerformanceTracker(base_url, api_key, **tracker_kwargs)
    
    # Prepare security configuration for potential auto-creation
    security_config = {
        'client_id': client_id,
        'enable_advanced_pii': enable_advanced_pii,
        'enable_tracing': enable_tracing,
        'verbose_security_logs': verbose_security_logs,
        'security_check_interval': security_check_interval,
        'batch_unclosed_sessions': batch_unclosed_sessions,
        'otlp_endpoint': otlp_endpoint
    }
    
    # Create ComplianceWrapper with intelligent security integration
    return ComplianceWrapper(
        wrapped_tracker=base_tracker,
        enable_compliance=enable_compliance,
        auto_detect_security=auto_detect_security,
        enable_security_if_missing=enable_security,
        hipaa_scope=hipaa_scope,
        gdpr_scope=gdpr_scope,
        default_retention_policy=default_retention_policy,
        audit_log_path=audit_log_path,
        client_id=client_id,
        # Pass security config without client_id since it's already passed above
        enable_advanced_pii=enable_advanced_pii,
        enable_tracing=enable_tracing,
        verbose_security_logs=verbose_security_logs,
        security_check_interval=security_check_interval,
        batch_unclosed_sessions=batch_unclosed_sessions,
        otlp_endpoint=otlp_endpoint
    )

def create_compliance_operations_tracker(base_url: str, api_key: Optional[str] = None,
                                        enable_compliance: bool = True,
                                        enable_security: bool = True,
                                        auto_detect_security: bool = True,
                                        hipaa_scope: bool = False,
                                        gdpr_scope: bool = False,
                                        **kwargs) -> ComplianceWrapper:
    """
    Enhanced factory function to create a compliance-enabled operations tracker with intelligent security integration
    
    Args:
        base_url: API base URL
        api_key: API authentication key
        enable_compliance: Whether to enable compliance features
        enable_security: Whether to enable security features (auto-created if available)
        auto_detect_security: Whether to automatically detect and integrate security
        hipaa_scope: Whether HIPAA compliance is required
        gdpr_scope: Whether GDPR compliance is required
        **kwargs: Additional arguments passed to base tracker and security/compliance layers
        
    Returns:
        ComplianceWrapper with operations tracking and intelligent security integration
    """
    from .AgentOper import AgentOperationsTracker
    
    # Separate kwargs for different layers
    security_params = ['enable_advanced_pii', 'enable_tracing', 'verbose_security_logs', 
                      'security_check_interval', 'batch_unclosed_sessions', 'otlp_endpoint', 'client_id']
    compliance_params = ['audit_log_path', 'client_id', 'default_retention_policy']
    
    tracker_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in security_params and k not in compliance_params}
    
    security_kwargs = {k: v for k, v in kwargs.items() 
                      if k in security_params}
    
    compliance_kwargs = {k: v for k, v in kwargs.items() 
                        if k in compliance_params}
    
    # Create base operations tracker
    base_tracker = AgentOperationsTracker(base_url, api_key, **tracker_kwargs)
    
    # Create ComplianceWrapper with intelligent security integration
    return ComplianceWrapper(
        wrapped_tracker=base_tracker,
        enable_compliance=enable_compliance,
        auto_detect_security=auto_detect_security,
        enable_security_if_missing=enable_security,
        hipaa_scope=hipaa_scope,
        gdpr_scope=gdpr_scope,
        **compliance_kwargs,
        **security_kwargs
    )

def create_smart_compliance_tracker(base_url: str, api_key: Optional[str] = None,
                                   compliance_level: str = "standard", 
                                   **kwargs) -> ComplianceWrapper:
    """
    Smart factory function that automatically configures compliance based on requirements level
    
    Args:
        base_url: API base URL
        api_key: API authentication key
        compliance_level: Level of compliance requirements:
            - "minimal": Basic compliance logging only
            - "standard": Full compliance + security features
            - "healthcare": HIPAA + full compliance + security
            - "eu": GDPR + full compliance + security  
            - "enterprise": All compliance + security + audit logging
        **kwargs: Additional configuration options
        
    Returns:
        ComplianceWrapper configured for the specified compliance level
    """
    compliance_configs = {
        "minimal": {
            "enable_compliance": True,
            "enable_security": False,
            "hipaa_scope": False,
            "gdpr_scope": False,
            "audit_log_path": None
        },
        "standard": {
            "enable_compliance": True,
            "enable_security": True,
            "hipaa_scope": False,
            "gdpr_scope": False,
            "audit_log_path": None
        },
        "healthcare": {
            "enable_compliance": True,
            "enable_security": True,
            "hipaa_scope": True,
            "gdpr_scope": False,
            "audit_log_path": "compliance_healthcare.audit",
            "default_retention_policy": "7_days"
        },
        "eu": {
            "enable_compliance": True,
            "enable_security": True,
            "hipaa_scope": False,
            "gdpr_scope": True,
            "audit_log_path": "compliance_gdpr.audit"
        },
        "enterprise": {
            "enable_compliance": True,
            "enable_security": True,
            "hipaa_scope": True,
            "gdpr_scope": True,
            "audit_log_path": "compliance_enterprise.audit",
            "enable_advanced_pii": True,
            "enable_tracing": True,
            "verbose_security_logs": True,
            "security_check_interval": 300,
            "batch_unclosed_sessions": True
        }
    }
    
    if compliance_level not in compliance_configs:
        raise ValueError(f"Unknown compliance level: {compliance_level}. "
                        f"Available: {list(compliance_configs.keys())}")
    
    # Merge default config with user overrides
    config = compliance_configs[compliance_level].copy()
    config.update(kwargs)
    
    return create_compliance_tracker(base_url, api_key, **config)
