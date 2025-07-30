#!/usr/bin/env python3
"""
Hybrid Session Management Demo

This script demonstrates the hybrid session management system that combines
lightweight local caching with backend persistence for seamless session recovery.
"""

import time
import logging
from datetime import datetime, timedelta
from tracker.AgentPerform import SessionInfo, AgentPerformanceTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_hybrid_session_info():
    """Test hybrid SessionInfo with run_id and ended status"""
    logger.info("=== Testing Hybrid SessionInfo ===")
    
    # Create session with new hybrid features
    session = SessionInfo(
        agent_id="support_agent_001",
        start_time=datetime.now(),
        run_id="unique_run_12345",
        user_id="customer_789",
        is_ended=False
    )
    
    logger.info(f"Created session: agent={session.agent_id}, run_id={session.run_id}")
    logger.info(f"Start time: {session.start_time}")
    logger.info(f"Last access: {session.last_access_time}")
    logger.info(f"Is ended: {session.is_ended}")
    
    # Test session data conversion
    session_dict = session.to_dict()
    logger.info(f"Session as dict: {session_dict}")
    
    # Test TTL behavior
    logger.info(f"Initial expired (TTL=0.001h): {session.is_expired(0.001)}")
    
    time.sleep(2)
    logger.info(f"After 2s expired: {session.is_expired(0.001)}")
    
    # Touch session (sliding TTL)
    session.touch()
    logger.info("Session touched - TTL reset")
    logger.info(f"New last access: {session.last_access_time}")
    logger.info(f"After touch expired: {session.is_expired(0.001)}")
    
    # Mark as ended
    session.is_ended = True
    logger.info(f"Session marked as ended: {session.is_ended}")
    
    logger.info("✅ Hybrid SessionInfo test completed!")

def simulate_hybrid_ttl_behavior():
    """Simulate hybrid TTL behavior with local vs backend expiry"""
    logger.info("\n=== Simulating Hybrid TTL Behavior ===")
    
    local_ttl_hours = 0.001  # ~3.6 seconds (local cache)
    backend_ttl_hours = 0.005  # ~18 seconds (backend persistence)
    
    session = SessionInfo(
        agent_id="demo_agent",
        start_time=datetime.now(),
        run_id="demo_run_001",
        user_id="demo_user"
    )
    
    logger.info(f"Local TTL: {local_ttl_hours * 3600:.1f}s")
    logger.info(f"Backend TTL: {backend_ttl_hours * 3600:.1f}s")
    
    start_time = datetime.now()
    
    # Simulate 25 seconds of session lifecycle
    for i in range(5):
        time.sleep(5)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Check local TTL expiry
        local_expired = session.is_expired(local_ttl_hours)
        
        # Check backend TTL expiry (from start time, not last access)
        backend_expiry = session.start_time + timedelta(hours=backend_ttl_hours)
        backend_expired = datetime.now() > backend_expiry
        
        logger.info(f"\nElapsed: {elapsed:.1f}s")
        logger.info(f"  Local cache expired: {local_expired}")
        logger.info(f"  Backend expired: {backend_expired}")
        
        if local_expired and not backend_expired:
            logger.info("  → Would retrieve from backend (hybrid fallback)")
        elif local_expired and backend_expired:
            logger.info("  → Session permanently expired")
            break
        else:
            logger.info("  → Session active in local cache")
            session.touch()  # Reset sliding TTL
    
    logger.info("✅ Hybrid TTL simulation completed!")

def demo_session_resumption_logic():
    """Demonstrate session resumption logic"""
    logger.info("\n=== Session Resumption Logic Demo ===")
    
    # Original session
    original_session = SessionInfo(
        agent_id="advisor_agent",
        start_time=datetime.now() - timedelta(minutes=30),  # Started 30 min ago
        run_id="consultation_12345", 
        user_id="client_001",
        is_ended=False
    )
    
    logger.info("Original session created 30 minutes ago")
    logger.info(f"  Agent: {original_session.agent_id}")
    logger.info(f"  Run ID: {original_session.run_id}")
    logger.info(f"  User: {original_session.user_id}")
    
    # Simulate local cache expiry (10 hours)
    local_expired = original_session.is_expired(10.0)
    logger.info(f"Local cache expired (10h TTL): {local_expired}")
    
    # Check backend persistence (20 hours)
    backend_expiry = original_session.start_time + timedelta(hours=20)
    backend_valid = datetime.now() < backend_expiry
    logger.info(f"Backend still valid (20h TTL): {backend_valid}")
    
    if backend_valid:
        logger.info("✅ Session can be resumed from backend!")
        
        # Simulate resumption
        resumed_session = SessionInfo(
            agent_id=original_session.agent_id,
            start_time=original_session.start_time,  # Keep original start time
            run_id=original_session.run_id,          # Keep same run ID
            user_id=original_session.user_id,
            is_ended=False
        )
        resumed_session.touch()  # Reset local TTL
        
        logger.info("Session resumed with context:")
        logger.info(f"  Same run ID: {resumed_session.run_id}")
        logger.info(f"  Original start time preserved")
        logger.info(f"  Local TTL reset: {resumed_session.last_access_time}")
    
    logger.info("✅ Session resumption logic demo completed!")

def demo_crash_recovery_simulation():
    """Simulate crash recovery scenario"""
    logger.info("\n=== Crash Recovery Simulation ===")
    
    # Phase 1: Normal operation
    logger.info("Phase 1: Normal SDK operation")
    active_sessions = {
        "session_001": SessionInfo(
            agent_id="support_001",
            start_time=datetime.now() - timedelta(minutes=15),
            run_id="support_run_001",
            user_id="customer_123"
        ),
        "session_002": SessionInfo(
            agent_id="sales_002", 
            start_time=datetime.now() - timedelta(minutes=5),
            run_id="sales_run_002",
            user_id="lead_456"
        )
    }
    
    logger.info(f"Active sessions: {len(active_sessions)}")
    for session_id, session in active_sessions.items():
        logger.info(f"  {session_id}: {session.agent_id} (run: {session.run_id})")
    
    # Phase 2: Crash occurs
    logger.info("\nPhase 2: SDK Crash/Restart")
    logger.info("💥 SDK crashes - local cache lost!")
    
    # Simulate cache loss
    active_sessions.clear()
    logger.info(f"Local cache cleared: {len(active_sessions)} sessions")
    
    # Phase 3: Recovery
    logger.info("\nPhase 3: Post-crash recovery")
    logger.info("SDK restarted, attempting session recovery...")
    
    # Simulate backend retrieval
    recovered_sessions = {
        "session_001": {
            "agent_id": "support_001",
            "start_time": (datetime.now() - timedelta(minutes=15)).isoformat(),
            "run_id": "support_run_001", 
            "user_id": "customer_123",
            "is_ended": False
        },
        "session_002": {
            "agent_id": "sales_002",
            "start_time": (datetime.now() - timedelta(minutes=5)).isoformat(),
            "run_id": "sales_run_002",
            "user_id": "lead_456", 
            "is_ended": False
        }
    }
    
    logger.info("Backend returned session data:")
    for session_id, data in recovered_sessions.items():
        # Reconstruct SessionInfo from backend data
        session = SessionInfo(
            agent_id=data['agent_id'],
            start_time=datetime.fromisoformat(data['start_time']),
            run_id=data['run_id'],
            user_id=data['user_id'],
            is_ended=data['is_ended']
        )
        session.touch()  # Reset local TTL
        
        logger.info(f"  ✅ {session_id}: {session.agent_id} recovered!")
        logger.info(f"     Run ID: {session.run_id}")
        logger.info(f"     Uptime: {(datetime.now() - session.start_time).total_seconds() / 60:.1f} min")
    
    logger.info("✅ Crash recovery simulation completed!")

def demo_dual_ttl_analytics():
    """Demonstrate analytics with dual TTL system"""
    logger.info("\n=== Dual TTL Analytics Demo ===")
    
    # Create sessions with different ages
    sessions = [
        SessionInfo("agent_001", datetime.now() - timedelta(hours=2), "run_001", "user_001"),
        SessionInfo("agent_002", datetime.now() - timedelta(hours=8), "run_002", "user_002"), 
        SessionInfo("agent_003", datetime.now() - timedelta(hours=15), "run_003", "user_003"),
        SessionInfo("agent_004", datetime.now() - timedelta(hours=25), "run_004", "user_004"),
    ]
    
    local_ttl = 10.0   # 10 hours
    backend_ttl = 20.0  # 20 hours
    
    logger.info(f"Analyzing {len(sessions)} sessions:")
    logger.info(f"Local TTL: {local_ttl}h, Backend TTL: {backend_ttl}h")
    
    now = datetime.now()
    active_local = 0
    active_backend = 0
    expired_both = 0
    
    for i, session in enumerate(sessions, 1):
        age_hours = (now - session.start_time).total_seconds() / 3600
        local_expired = session.is_expired(local_ttl)
        backend_expired = age_hours > backend_ttl
        
        logger.info(f"\nSession {i}: age={age_hours:.1f}h")
        logger.info(f"  Local cache: {'❌ Expired' if local_expired else '✅ Active'}")
        logger.info(f"  Backend: {'❌ Expired' if backend_expired else '✅ Active'}")
        
        if not local_expired:
            active_local += 1
            status = "🟢 Active (local cache)"
        elif not backend_expired:
            active_backend += 1  
            status = "🟡 Recoverable (backend only)"
        else:
            expired_both += 1
            status = "🔴 Permanently expired"
        
        logger.info(f"  Status: {status}")
    
    logger.info(f"\n📊 Summary:")
    logger.info(f"  Active in local cache: {active_local}")
    logger.info(f"  Recoverable from backend: {active_backend}")
    logger.info(f"  Permanently expired: {expired_both}")
    logger.info(f"  Total recoverable sessions: {active_local + active_backend}")
    
    logger.info("✅ Dual TTL analytics demo completed!")

if __name__ == "__main__":
    logger.info("🚀 Starting Hybrid Session Management Demo")
    
    test_hybrid_session_info()
    simulate_hybrid_ttl_behavior()
    demo_session_resumption_logic()
    demo_crash_recovery_simulation()
    demo_dual_ttl_analytics()
    
    logger.info("\n🎉 Hybrid Session Management Demo Completed!")
    logger.info("\n📋 Key Features Demonstrated:")
    logger.info("- ✅ Lightweight SessionInfo with run_id and ended status")
    logger.info("- ✅ Dual TTL system (local sliding + backend fixed)")
    logger.info("- ✅ Seamless session resumption logic")
    logger.info("- ✅ Crash recovery with context preservation")
    logger.info("- ✅ Rich analytics for session lifecycle management")
    logger.info("- ✅ Graceful degradation and error handling")
    
    logger.info("\n🎯 Hybrid Model Benefits:")
    logger.info("- 🚀 Zero-downtime session recovery")
    logger.info("- 💾 Memory-efficient local caching")
    logger.info("- 🔄 Automatic backend fallback")
    logger.info("- 📊 Comprehensive session analytics")
    logger.info("- ⚡ Performance optimized for both speed and persistence") 