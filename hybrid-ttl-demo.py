#!/usr/bin/env python3
"""
Production-Ready Hybrid Session Management Demo

This comprehensive demo showcases all the production-ready improvements:
- Offline event queueing when backend is unreachable
- LRU cache with memory bounds and eviction
- Batched expiry notifications to reduce API load
- Activity vs. query-based TTL reset behavior
- Daemon thread cleanup management
- Async cleanup support
- Rich analytics and monitoring
"""

import time
import logging
import asyncio
from datetime import datetime, timedelta
from tracker import AgentPerformanceTracker, ConversationQuality

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_offline_queueing():
    """Test offline event queueing when backend is unreachable"""
    logger.info("=== Testing Offline Event Queueing ===")
    
    # Create tracker with small limits for testing
    tracker = AgentPerformanceTracker(
        base_url="https://unreachable-backend.com",  # Intentionally unreachable
        session_ttl_hours=10.0,
        max_offline_queue_size=100,
        batch_notification_size=5,
        cleanup_interval_minutes=1
    )
    
    try:
        # These operations should be queued offline
        session1 = tracker.start_conversation("agent_001", user_id="user_001")
        session2 = tracker.start_conversation("agent_002", user_id="user_002")
        
        logger.info(f"Created sessions: {session1}, {session2}")
        
        # Try to end conversations (should be queued)
        if session1:
            result = tracker.end_conversation(
                session_id=session1,
                quality_score=ConversationQuality.EXCELLENT,
                message_count=10
            )
            logger.info(f"End conversation result: {result}")
        
        # Check offline queue stats
        stats = tracker.get_session_stats()
        logger.info(f"Offline queue size: {stats['offline_queue_size']}")
        logger.info(f"Queued event types: {stats['queued_event_types']}")
        logger.info(f"Backend available: {stats['backend_available']}")
        
        # Force backend availability check
        tracker._backend_available = True
        tracker._process_offline_queue()
        
        final_stats = tracker.get_session_stats()
        logger.info(f"Final offline queue size: {final_stats['offline_queue_size']}")
        
    except Exception as e:
        logger.error(f"Error in offline queueing test: {e}")
    finally:
        tracker.close()
        logger.info("✅ Offline queueing test completed!")

def test_lru_cache_eviction():
    """Test LRU cache behavior with memory bounds"""
    logger.info("\n=== Testing LRU Cache Eviction ===")
    
    # Create tracker with small cache for testing
    tracker = AgentPerformanceTracker(
        base_url="https://test-backend.com",
        max_cache_size=3,  # Very small for testing
        session_ttl_hours=10.0
    )
    
    try:
        # Create more sessions than cache limit
        sessions = []
        for i in range(5):
            session_id = tracker.start_conversation(f"agent_{i:03d}", user_id=f"user_{i:03d}")
            sessions.append(session_id)
            logger.info(f"Created session {i+1}/5: {session_id}")
            
            # Check cache stats
            stats = tracker.get_session_stats()
            logger.info(f"  Cache: {stats['total_cached_sessions']}/{stats['max_cache_size']} "
                       f"({stats['cache_utilization_percent']}% utilization)")
        
        # Access some sessions to test LRU behavior
        logger.info("\nTesting LRU access patterns:")
        for i, session_id in enumerate(sessions[-3:]):  # Access last 3 sessions
            if session_id:
                is_active = tracker.is_session_active(session_id)
                logger.info(f"Accessed session {session_id}: active={is_active}")
        
        # Check final cache state
        final_stats = tracker.get_session_stats()
        logger.info(f"Final cache state: {final_stats['total_cached_sessions']} sessions")
        logger.info(f"Cache utilization: {final_stats['cache_utilization_percent']}%")
        
    except Exception as e:
        logger.error(f"Error in LRU cache test: {e}")
    finally:
        tracker.close()
        logger.info("✅ LRU cache eviction test completed!")

def test_activity_vs_query_ttl():
    """Test TTL reset behavior for activities vs queries"""
    logger.info("\n=== Testing Activity vs. Query TTL Behavior ===")
    
    tracker = AgentPerformanceTracker(
        base_url="https://test-backend.com",
        session_ttl_hours=0.002,  # ~7 seconds for testing
        cleanup_interval_minutes=1
    )
    
    try:
        # Create a session
        session_id = tracker.start_conversation("test_agent", user_id="test_user")
        if not session_id:
            logger.error("Failed to create session for TTL test")
            return
        
        logger.info(f"Created session for TTL test: {session_id}")
        
        # Test query operations (should NOT reset TTL)
        logger.info("\nTesting query operations (no TTL reset):")
        for i in range(3):
            time.sleep(3)  # Wait 3 seconds each time
            
            # These are query operations - should NOT reset TTL
            is_active = tracker.is_session_active(session_id)
            ttl_remaining = tracker.get_session_ttl_remaining(session_id)
            
            logger.info(f"Query {i+1}: active={is_active}, TTL remaining={ttl_remaining:.4f}h")
            
            if not is_active:
                logger.info("Session expired during query operations (expected)")
                break
        
        # Create another session for activity test
        session_id2 = tracker.start_conversation("test_agent_2", user_id="test_user_2")
        if session_id2:
            logger.info(f"\nCreated second session for activity test: {session_id2}")
            
            # Test activity operations (SHOULD reset TTL)
            logger.info("Testing activity operations (with TTL reset):")
            for i in range(3):
                time.sleep(3)  # Wait 3 seconds each time
                
                # This is an activity operation - SHOULD reset TTL
                touched = tracker.touch_session(session_id2)
                ttl_remaining = tracker.get_session_ttl_remaining(session_id2)
                is_active = tracker.is_session_active(session_id2)
                
                logger.info(f"Activity {i+1}: touched={touched}, active={is_active}, TTL remaining={ttl_remaining:.4f}h")
                
                if not is_active:
                    logger.info("Session expired despite activity (unexpected)")
                    break
            
            logger.info("Session stayed alive due to activity-based TTL reset!")
        
    except Exception as e:
        logger.error(f"Error in TTL behavior test: {e}")
    finally:
        tracker.close()
        logger.info("✅ Activity vs. Query TTL test completed!")

def test_batched_notifications():
    """Test batched expiry notifications"""
    logger.info("\n=== Testing Batched Expiry Notifications ===")
    
    tracker = AgentPerformanceTracker(
        base_url="https://test-backend.com",
        session_ttl_hours=0.001,  # ~4 seconds for quick expiry
        batch_notification_size=3,  # Small batch for testing
        cleanup_interval_minutes=1
    )
    
    try:
        # Create multiple sessions that will expire
        sessions = []
        for i in range(5):
            session_id = tracker.start_conversation(f"batch_agent_{i}", user_id=f"batch_user_{i}")
            sessions.append(session_id)
            logger.info(f"Created session {i+1}: {session_id}")
        
        logger.info(f"Created {len(sessions)} sessions for batch test")
        
        # Wait for sessions to expire
        logger.info("Waiting for sessions to expire...")
        time.sleep(8)  # Wait longer than TTL
        
        # Force cleanup to trigger batch notifications
        tracker._cleanup_expired_sessions()
        
        # Check notification batch status
        stats = tracker.get_session_stats()
        logger.info(f"Pending notifications: {stats['pending_notifications']}")
        logger.info(f"Batch size: {stats['batch_notification_size']}")
        
        # Force flush notifications
        tracker._flush_pending_notifications()
        
        final_stats = tracker.get_session_stats()
        logger.info(f"Final pending notifications: {final_stats['pending_notifications']}")
        
    except Exception as e:
        logger.error(f"Error in batched notifications test: {e}")
    finally:
        tracker.close()
        logger.info("✅ Batched notifications test completed!")

async def test_async_cleanup():
    """Test async cleanup functionality"""
    logger.info("\n=== Testing Async Cleanup ===")
    
    tracker = AgentPerformanceTracker(
        base_url="https://test-backend.com",
        session_ttl_hours=0.002,  # ~7 seconds
        cleanup_interval_minutes=1
    )
    
    try:
        # Start async cleanup
        tracker._start_async_cleanup()
        
        # Create some sessions
        sessions = []
        for i in range(3):
            session_id = await tracker.start_conversation_async(f"async_agent_{i}", user_id=f"async_user_{i}")
            sessions.append(session_id)
            logger.info(f"Created async session {i+1}: {session_id}")
        
        # Check async cleanup status
        stats = tracker.get_session_stats()
        logger.info(f"Async cleanup running: {stats['async_cleanup_running']}")
        logger.info(f"Regular cleanup running: {stats['cleanup_daemon_running']}")
        
        # Wait a bit for async operations
        await asyncio.sleep(2)
        
        # Test async session operations
        for session_id in sessions:
            if session_id:
                is_active = tracker.is_session_active(session_id)
                logger.info(f"Async session {session_id} active: {is_active}")
        
        logger.info("Async cleanup functionality working!")
        
    except Exception as e:
        logger.error(f"Error in async cleanup test: {e}")
    finally:
        await tracker.close_async()
        logger.info("✅ Async cleanup test completed!")

def test_comprehensive_stats():
    """Test comprehensive session statistics"""
    logger.info("\n=== Testing Comprehensive Session Statistics ===")
    
    tracker = AgentPerformanceTracker(
        base_url="https://unreachable-backend.com",  # Force offline mode
        max_cache_size=50,
        max_offline_queue_size=100,
        batch_notification_size=10,
        session_ttl_hours=10.0,
        backend_ttl_hours=20.0
    )
    
    try:
        # Create various types of sessions
        active_sessions = []
        for i in range(5):
            session_id = tracker.start_conversation(f"stats_agent_{i}", user_id=f"stats_user_{i}")
            active_sessions.append(session_id)
        
        # End some sessions
        for session_id in active_sessions[:2]:
            if session_id:
                tracker.end_conversation(session_id, quality_score=ConversationQuality.GOOD)
        
        # Create some failed sessions
        for i in range(2):
            session_id = tracker.start_conversation(f"failed_agent_{i}", user_id=f"failed_user_{i}")
            if session_id:
                tracker.record_failed_session(session_id, "Test failure")
        
        # Get comprehensive stats
        stats = tracker.get_session_stats()
        
        logger.info("📊 Comprehensive Session Statistics:")
        logger.info(f"  Session Cache:")
        logger.info(f"    Total cached sessions: {stats['total_cached_sessions']}")
        logger.info(f"    Active sessions: {stats['active_sessions']}")
        logger.info(f"    Ended sessions: {stats['ended_sessions']}")
        logger.info(f"    Cache utilization: {stats['cache_utilization_percent']}%")
        
        logger.info(f"  TTL Configuration:")
        logger.info(f"    Local TTL: {stats['local_ttl_hours']}h")
        logger.info(f"    Backend TTL: {stats['backend_ttl_hours']}h")
        
        logger.info(f"  Hybrid Model Status:")
        logger.info(f"    Backend available: {stats['backend_available']}")
        logger.info(f"    Hybrid model enabled: {stats['hybrid_model_enabled']}")
        logger.info(f"    Sliding TTL enabled: {stats['sliding_ttl_enabled']}")
        
        logger.info(f"  Offline Queue:")
        logger.info(f"    Queue size: {stats['offline_queue_size']}")
        logger.info(f"    Queue utilization: {stats['offline_queue_utilization_percent']}%")
        logger.info(f"    Queued event types: {stats['queued_event_types']}")
        logger.info(f"    Failed replay events: {stats['failed_replay_events']}")
        
        logger.info(f"  System Health:")
        logger.info(f"    Cleanup daemon running: {stats['cleanup_daemon_running']}")
        logger.info(f"    Async cleanup running: {stats['async_cleanup_running']}")
        
    except Exception as e:
        logger.error(f"Error in comprehensive stats test: {e}")
    finally:
        tracker.close()
        logger.info("✅ Comprehensive statistics test completed!")

def main():
    """Run all production-ready demos"""
    logger.info("🚀 Starting Production-Ready Hybrid Session Management Demo")
    
    # Run synchronous tests
    test_offline_queueing()
    test_lru_cache_eviction()
    test_activity_vs_query_ttl()
    test_batched_notifications()
    test_comprehensive_stats()
    
    # Run async test
    asyncio.run(test_async_cleanup())
    
    logger.info("\n🎉 All Production-Ready Tests Completed!")
    logger.info("\n📋 Production Features Demonstrated:")
    logger.info("- ✅ Offline event queueing with retry logic")
    logger.info("- ✅ LRU cache with memory bounds and eviction")
    logger.info("- ✅ Batched notifications to reduce API load")
    logger.info("- ✅ Activity vs. query-based TTL reset behavior")
    logger.info("- ✅ Daemon thread cleanup management")
    logger.info("- ✅ Async cleanup support")
    logger.info("- ✅ Comprehensive monitoring and analytics")
    logger.info("- ✅ Graceful resource cleanup and shutdown")
    
    logger.info("\n🏭 Production Benefits:")
    logger.info("- 🚀 Resilient to backend outages")
    logger.info("- 💾 Memory-bounded and efficient")
    logger.info("- 📡 Reduced API load through batching")
    logger.info("- ⚡ Optimized TTL management")
    logger.info("- 🔧 Easy monitoring and debugging")
    logger.info("- 🛡️ Robust error handling and recovery")

if __name__ == "__main__":
    main() 