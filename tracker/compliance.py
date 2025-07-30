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
    """
    Manages compliance tracking, audit logging, and policy violations
    
    Provides immutable audit trail for SOC 2, ISO 27001, HIPAA, and GDPR compliance
    """
    
    def __init__(self, client_id: str, 
                 hipaa_scope: bool = False,
                 gdpr_scope: bool = False,
                 default_retention_policy: str = "30_days",
                 audit_log_path: Optional[str] = None,
                 enable_hash_chaining: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Compliance Manager
        
        Args:
            client_id: Unique client identifier
            hipaa_scope: Whether HIPAA compliance is required
            gdpr_scope: Whether GDPR compliance is required
            default_retention_policy: Default data retention policy
            audit_log_path: Path for audit log file (optional)
            enable_hash_chaining: Enable tamper detection via hash chaining
            logger: Optional logger instance
        """
        self.client_id = client_id
        self.hipaa_scope = hipaa_scope
        self.gdpr_scope = gdpr_scope
        self.default_retention_policy = default_retention_policy
        self.enable_hash_chaining = enable_hash_chaining
        self.logger = logger or logging.getLogger(__name__)
        
        # Thread safety
        self._audit_lock = threading.RLock()
        self._acknowledgment_lock = threading.RLock()
        
        # Immutable audit log (append-only)
        self._audit_log: List[ComplianceAuditEntry] = []
        self._last_entry_hash: Optional[str] = None
        
        # User acknowledgments (persistent across sessions)
        self._acknowledgments: Dict[str, List[UserAcknowledgment]] = {}
        
        # Detectors
        self.phi_detector = PHIDetector()
        self.gdpr_detector = GDPRDetector()
        
        # Audit log file (for persistence)
        self.audit_log_path = audit_log_path
        if audit_log_path:
            self._ensure_audit_log_file()
        
        self.logger.info(f"ComplianceManager initialized for {client_id} (HIPAA: {hipaa_scope}, GDPR: {gdpr_scope})")
    
    def _ensure_audit_log_file(self):
        """Ensure audit log file exists and is properly initialized"""
        if not self.audit_log_path:
            return
        
        try:
            log_path = Path(self.audit_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file if it doesn't exist
            if not log_path.exists():
                with open(log_path, 'w') as f:
                    f.write(f"# Compliance Audit Log - {self.client_id}\n")
                    f.write(f"# Created: {datetime.now().isoformat()}\n")
                    f.write("# Format: JSON Lines (one entry per line)\n\n")
                
                self.logger.info(f"Created audit log file: {log_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to create audit log file: {e}")
    
    def _append_to_audit_file(self, entry: ComplianceAuditEntry):
        """Append entry to persistent audit log file"""
        if not self.audit_log_path:
            return
        
        try:
            with open(self.audit_log_path, 'a') as f:
                json.dump(entry.to_dict(), f, separators=(',', ':'))
                f.write('\n')
        
        except Exception as e:
            self.logger.error(f"Failed to write to audit log file: {e}")
    
    def _detect_policy_violations(self, session_id: str, 
                                 text_content: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None,
                                 compliance_flags: Optional[ComplianceFlags] = None) -> List[PolicyViolation]:
        """
        Detect policy violations based on content and flags
        
        Args:
            session_id: Session identifier
            text_content: Text content to analyze
            metadata: Session metadata
            compliance_flags: Current compliance flags
            
        Returns:
            List of detected policy violations
        """
        violations = []
        
        if not compliance_flags:
            return violations
        
        # Check for acknowledged violations (skip if already acknowledged)
        acknowledged_types = self._get_acknowledged_violation_types(session_id)
        
        # HIPAA Violations
        if self.hipaa_scope and compliance_flags.phi_detected:
            # Check if PHI is being sent to non-HIPAA backends
            non_hipaa_backends = []
            for backend in compliance_flags.data_sent_to:
                if not self._is_hipaa_compliant_backend(backend):
                    non_hipaa_backends.append(backend)
            
            if non_hipaa_backends and "hipaa_violation" not in acknowledged_types:
                violations.append(PolicyViolation(
                    type="hipaa_violation",
                    details=f"PHI detected but sent to non-HIPAA compliant backends: {non_hipaa_backends}",
                    severity="high"
                ))
        
        # GDPR Violations
        if compliance_flags.gdpr_scope and compliance_flags.pii_detected:
            # Check cross-border data transfer
            if (compliance_flags.user_region and 
                compliance_flags.data_processing_region and
                self.gdpr_detector.is_eu_region(compliance_flags.user_region) and
                not self.gdpr_detector.is_eu_region(compliance_flags.data_processing_region)):
                
                if "gdpr_violation" not in acknowledged_types:
                    violations.append(PolicyViolation(
                        type="gdpr_violation", 
                        details=f"EU user data processed outside EU: {compliance_flags.data_processing_region}",
                        severity="high"
                    ))
        
        # Data Retention Violations
        if compliance_flags.retention_policy == "7_days" and metadata:
            # Check if session is older than retention period
            session_age = self._calculate_session_age(session_id, metadata)
            if session_age and session_age > 7:
                if "retention_violation" not in acknowledged_types:
                    violations.append(PolicyViolation(
                        type="retention_violation",
                        details=f"Session data retained beyond policy: {session_age} days > 7 days",
                        severity="medium"
                    ))
        
        return violations
    
    def _is_hipaa_compliant_backend(self, backend: str) -> bool:
        """Check if backend is HIPAA compliant (configurable)"""
        # This would typically be configured based on your specific backends
        hipaa_compliant_backends = {
            'hipaa-api.example.com',
            'secure-health.example.com',
            'medical-ai.example.com'
        }
        
        return any(compliant in backend.lower() for compliant in hipaa_compliant_backends)
    
    def _calculate_session_age(self, session_id: str, metadata: Dict[str, Any]) -> Optional[int]:
        """Calculate session age in days"""
        try:
            if 'start_time' in metadata:
                start_time = datetime.fromisoformat(metadata['start_time'].replace('Z', '+00:00'))
                age = (datetime.now() - start_time).days
                return age
        except Exception:
            pass
        return None
    
    def _get_acknowledged_violation_types(self, session_id: str) -> Set[str]:
        """Get set of violation types that have been acknowledged for this session"""
        acknowledged_types = set()
        
        with self._acknowledgment_lock:
            if session_id in self._acknowledgments:
                for ack in self._acknowledgments[session_id]:
                    acknowledged_types.add(ack.policy_type)
        
        return acknowledged_types
    
    def create_compliance_flags(self, text_content: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None,
                              user_region: Optional[str] = None,
                              data_sent_to: Optional[List[str]] = None) -> ComplianceFlags:
        """
        Create compliance flags based on session content and context
        
        Args:
            text_content: Text content to analyze
            metadata: Session metadata
            user_region: User's region code
            data_sent_to: List of backend services data is sent to
            
        Returns:
            ComplianceFlags object with detected compliance requirements
        """
        flags = ComplianceFlags(
            gdpr_scope=self.gdpr_scope,
            hipaa_scope=self.hipaa_scope,
            retention_policy=self.default_retention_policy,
            user_region=user_region,
            data_sent_to=data_sent_to or []
        )
        
        # Detect PII (reuse from security module if available)
        try:
            from .security import detect_pii, scan_metadata_for_pii
            
            if text_content:
                flags.pii_detected = detect_pii(text_content)
            
            if metadata:
                flags.pii_detected = flags.pii_detected or scan_metadata_for_pii(metadata)
        
        except ImportError:
            # Fallback basic PII detection
            if text_content:
                flags.pii_detected = self._basic_pii_detection(text_content)
        
        # Detect PHI for HIPAA
        if text_content and self.hipaa_scope:
            flags.phi_detected = self.phi_detector.detect_phi(text_content)
        
        # GDPR scope detection
        if not flags.gdpr_scope:  # Only override if not explicitly set
            flags.gdpr_scope = self.gdpr_detector.detect_gdpr_scope(metadata, user_region)
        
        return flags
    
    def _basic_pii_detection(self, text: str) -> bool:
        """Basic PII detection fallback"""
        pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'  # Phone
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def log_compliance_event(self, session_id: str, event_type: str,
                           compliance_flags: ComplianceFlags,
                           text_content: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a compliance event to the immutable audit trail
        
        Args:
            session_id: Session identifier
            event_type: Type of event (start_conversation, end_conversation, etc.)
            compliance_flags: Compliance flags for this event
            text_content: Optional text content for violation detection
            metadata: Optional session metadata
            
        Returns:
            str: Entry ID of the logged audit entry
        """
        with self._audit_lock:
            # Generate unique entry ID
            entry_id = f"audit_{uuid.uuid4().hex[:12]}"
            
            # Detect policy violations
            violations = self._detect_policy_violations(
                session_id, text_content, metadata, compliance_flags
            )
            
            # Check if user has acknowledged any violations
            user_acknowledged = len(self._get_acknowledged_violation_types(session_id)) > 0
            
            # Get acknowledgments for this session
            acknowledgments = self._acknowledgments.get(session_id, [])
            
            # Create audit entry
            entry = ComplianceAuditEntry(
                entry_id=entry_id,
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                event_type=event_type,
                compliance_flags=compliance_flags,
                policy_violations=violations,
                user_acknowledged=user_acknowledged,
                acknowledgments=acknowledgments.copy(),
                metadata=metadata or {},
                previous_hash=self._last_entry_hash
            )
            
            # Calculate hash for tamper detection
            if self.enable_hash_chaining:
                entry.entry_hash = entry.calculate_hash()
                self._last_entry_hash = entry.entry_hash
            
            # Append to immutable log
            self._audit_log.append(entry)
            
            # Persist to file if configured
            self._append_to_audit_file(entry)
            
            # Log violations if any
            if violations:
                violation_types = [v.type for v in violations]
                self.logger.warning(f"Compliance violations detected for session {session_id}: {violation_types}")
            
            self.logger.info(f"Compliance event logged: {event_type} for session {session_id} (violations: {len(violations)})")
            
            return entry_id
    
    def acknowledge_risk(self, session_id: str, policy_type: str, 
                        acknowledged_by: str, reason: str,
                        scope: str = "session") -> bool:
        """
        Allow user to acknowledge and accept specific policy risks
        
        Args:
            session_id: Session identifier
            policy_type: Type of policy violation being acknowledged
            acknowledged_by: User/system acknowledging the risk
            reason: Reason for the acknowledgment
            scope: Scope of acknowledgment (session, user, global)
            
        Returns:
            bool: True if acknowledgment was recorded
        """
        try:
            with self._acknowledgment_lock:
                # Create acknowledgment record
                acknowledgment = UserAcknowledgment(
                    session_id=session_id,
                    policy_type=policy_type,
                    acknowledged_by=acknowledged_by,
                    acknowledged_at=datetime.now().isoformat(),
                    reason=reason,
                    scope=scope
                )
                
                # Store acknowledgment
                if session_id not in self._acknowledgments:
                    self._acknowledgments[session_id] = []
                
                self._acknowledgments[session_id].append(acknowledgment)
                
                # Log acknowledgment event
                self.log_compliance_event(
                    session_id=session_id,
                    event_type="risk_acknowledgment",
                    compliance_flags=ComplianceFlags(),  # Empty flags for acknowledgment
                    metadata={
                        "acknowledgment": acknowledgment.to_dict(),
                        "scope": scope
                    }
                )
                
                self.logger.info(f"Risk acknowledged for session {session_id}: {policy_type} by {acknowledged_by}")
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to record risk acknowledgment: {e}")
            return False
    
    def get_audit_trail(self, session_id: Optional[str] = None,
                       event_type: Optional[str] = None,
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve audit trail entries with optional filtering
        
        Args:
            session_id: Filter by session ID
            event_type: Filter by event type
            start_time: Filter by start time (ISO format)
            end_time: Filter by end time (ISO format)
            
        Returns:
            List of audit entries as dictionaries
        """
        with self._audit_lock:
            entries = self._audit_log.copy()
        
        # Apply filters
        if session_id:
            entries = [e for e in entries if e.session_id == session_id]
        
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            entries = [e for e in entries if datetime.fromisoformat(e.timestamp) >= start_dt]
        
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            entries = [e for e in entries if datetime.fromisoformat(e.timestamp) <= end_dt]
        
        return [entry.to_dict() for entry in entries]
    
    def verify_audit_integrity(self) -> Dict[str, Any]:
        """
        Verify integrity of audit trail using hash chain
        
        Returns:
            Dict with integrity verification results
        """
        if not self.enable_hash_chaining:
            return {"integrity_enabled": False, "message": "Hash chaining disabled"}
        
        with self._audit_lock:
            entries = self._audit_log.copy()
        
        if not entries:
            return {"integrity_verified": True, "total_entries": 0}
        
        verification_results = {
            "integrity_verified": True,
            "total_entries": len(entries),
            "corrupted_entries": [],
            "hash_chain_intact": True
        }
        
        prev_hash = None
        for i, entry in enumerate(entries):
            # Verify entry hash
            expected_hash = entry.calculate_hash()
            if entry.entry_hash != expected_hash:
                verification_results["integrity_verified"] = False
                verification_results["corrupted_entries"].append({
                    "index": i,
                    "entry_id": entry.entry_id,
                    "issue": "Hash mismatch"
                })
            
            # Verify hash chain
            if entry.previous_hash != prev_hash:
                verification_results["integrity_verified"] = False
                verification_results["hash_chain_intact"] = False
                verification_results["corrupted_entries"].append({
                    "index": i,
                    "entry_id": entry.entry_id,
                    "issue": "Hash chain broken"
                })
            
            prev_hash = entry.entry_hash
        
        return verification_results
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """
        Get summary of compliance status and statistics
        
        Returns:
            Dict with compliance summary
        """
        with self._audit_lock:
            entries = self._audit_log.copy()
        
        # Count violations by type
        violation_counts = {}
        acknowledged_violations = 0
        total_sessions = set()
        
        for entry in entries:
            total_sessions.add(entry.session_id)
            
            for violation in entry.policy_violations:
                violation_type = violation.type
                violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
            
            if entry.user_acknowledged:
                acknowledged_violations += 1
        
        return {
            "client_id": self.client_id,
            "hipaa_scope": self.hipaa_scope,
            "gdpr_scope": self.gdpr_scope,
            "total_audit_entries": len(entries),
            "total_sessions_tracked": len(total_sessions),
            "policy_violations": violation_counts,
            "acknowledged_violations": acknowledged_violations,
            "hash_chaining_enabled": self.enable_hash_chaining,
            "last_entry_time": entries[-1].timestamp if entries else None
        }

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
                self.compliance_manager = ComplianceManager(
                    client_id=self.client_id,
                    hipaa_scope=hipaa_scope,
                    gdpr_scope=gdpr_scope,
                    default_retention_policy=default_retention_policy,
                    audit_log_path=audit_log_path,
                    logger=self.logger
                )
                
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
    
    def start_conversation(self, agent_id: str, user_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          user_region: Optional[str] = None) -> Optional[str]:
        """
        Start conversation with automatic compliance logging and policy violation detection
        
        Args:
            agent_id: Unique identifier for the AI agent
            user_id: Optional user identifier for the conversation
            metadata: Optional session metadata (may contain region, backend info)
            user_region: Optional explicit user region code (e.g., 'DE', 'US', 'FR')
            
        Returns:
            Optional[str]: Session ID if conversation started successfully, None otherwise
            
        Compliance Features:
            - Automatically detects GDPR scope based on user_region and metadata
            - Identifies data backends for compliance monitoring
            - Logs compliance event to immutable audit trail
            - Detects policy violations (HIPAA, GDPR, retention)
            
        Raises:
            Exception: Re-raises any exceptions from the wrapped tracker
        """
        try:
            # Call wrapped tracker
            session_id = self.wrapped_tracker.start_conversation(agent_id, user_id, metadata)
            
            # Log compliance event
            if session_id and self.enable_compliance and self.compliance_manager:
                # Determine data backends
                data_sent_to = self._detect_data_backends(metadata)
                
                # Create compliance flags
                compliance_flags = self.compliance_manager.create_compliance_flags(
                    text_content=None,  # No text content at start
                    metadata=metadata,
                    user_region=user_region,
                    data_sent_to=data_sent_to
                )
                
                # Log compliance event
                self.compliance_manager.log_compliance_event(
                    session_id=session_id,
                    event_type="start_conversation",
                    compliance_flags=compliance_flags,
                    metadata=metadata
                )
            
            return session_id
        
        except Exception as e:
            self.logger.error(f"Error in compliance start_conversation: {e}")
            raise
    
    def end_conversation(self, session_id: str, quality_score: Optional[Union[int, float]] = None,
                        user_feedback: Optional[str] = None,
                        message_count: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        End conversation with compliance logging and PII/PHI detection in user feedback
        
        Args:
            session_id: Unique session identifier from start_conversation
            quality_score: Optional conversation quality score (1-5 or 1.0-5.0)
            user_feedback: Optional user feedback text (scanned for PII/PHI)
            message_count: Optional total number of messages in conversation
            metadata: Optional session metadata for compliance analysis
            
        Returns:
            bool: True if conversation ended successfully, False otherwise
            
        Compliance Features:
            - Scans user_feedback for PII and PHI content
            - Detects policy violations based on detected sensitive data
            - Updates compliance flags with detected violations
            - Logs compliance event to immutable audit trail
            - Honors user acknowledgments to prevent duplicate violations
            
        Raises:
            Exception: Re-raises any exceptions from the wrapped tracker
        """
        try:
            # Call wrapped tracker
            result = self.wrapped_tracker.end_conversation(session_id, quality_score, user_feedback, message_count, metadata)
            
            # Log compliance event
            if result and self.enable_compliance and self.compliance_manager:
                # Determine data backends
                data_sent_to = self._detect_data_backends(metadata)
                
                # Create compliance flags
                compliance_flags = self.compliance_manager.create_compliance_flags(
                    text_content=user_feedback,
                    metadata=metadata,
                    data_sent_to=data_sent_to
                )
                
                # Log compliance event
                self.compliance_manager.log_compliance_event(
                    session_id=session_id,
                    event_type="end_conversation",
                    compliance_flags=compliance_flags,
                    text_content=user_feedback,
                    metadata=metadata
                )
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error in compliance end_conversation: {e}")
            raise
    
    def record_failed_session(self, session_id: str, error_message: str,
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
            result = self.wrapped_tracker.record_failed_session(session_id, error_message, metadata)
            
            # Log compliance event
            if result and self.enable_compliance and self.compliance_manager:
                # Determine data backends
                data_sent_to = self._detect_data_backends(metadata)
                
                # Create compliance flags
                compliance_flags = self.compliance_manager.create_compliance_flags(
                    text_content=error_message,
                    metadata=metadata,
                    data_sent_to=data_sent_to
                )
                
                # Log compliance event
                self.compliance_manager.log_compliance_event(
                    session_id=session_id,
                    event_type="record_failed_session",
                    compliance_flags=compliance_flags,
                    text_content=error_message,
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
            return self.compliance_manager.acknowledge_risk(
                session_id=session_id,
                policy_type=policy_type,
                acknowledged_by=acknowledged_by,
                reason=reason
            )
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
            summary = self.compliance_manager.get_compliance_summary()
            summary["security_integration"] = self.get_security_info()
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
            return self.compliance_manager.get_audit_trail(session_id=session_id)
        
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
            return self.compliance_manager.verify_audit_integrity()
        
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
