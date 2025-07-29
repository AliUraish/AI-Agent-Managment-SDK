#!/usr/bin/env python3
"""
TTL Session Management Demo

This script demonstrates the TTL-based session cleanup functionality
of the AgentPerformanceTracker.
"""

import time
import logging
import os
from tracker import AgentPerformanceTracker, ConversationQuality

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ttl_demo():
    """Demonstrate TTL session management"""
    logger.info("=== TTL Session Management Demo ===")
    
    # Initialize with short TTL for demo purposes (0.1 hours = 6 minutes)
    perf_tracker = AgentPerformanceTracker(
        base_url="https://your-backend-api.com",
        api_key=os.getenv('AGENT_TRACKER_API_KEY'),
        session_ttl_hours=0.1,  # 6 minutes for demo
        cleanup_interval_minutes=1,  # Check every minute
        logger=logger
    )
    
    try:
        logger.info("Starting multiple conversations...")
        
        # Start several conversations
        sessions = []
        for i in range(3):
            session_id = perf_tracker.start_conversation(
                agent_id=f"demo_agent_{i}",
                user_id=f"user_{i}",
                metadata={"demo": True, "batch": 1}
            )
            if session_id:
                sessions.append(session_id)
                logger.info(f"Started session {i+1}: {session_id}")
        
        # Check initial session stats
        stats = perf_tracker.get_session_stats()
        logger.info(f"Initial session stats: {stats}")
        
        # End one conversation normally
        if sessions:
            perf_tracker.end_conversation(
                session_id=sessions[0],
                quality_score=ConversationQuality.GOOD,
                message_count=10
            )
            logger.info(f"Ended session normally: {sessions[0]}")
        
        # Check stats after ending one session
        stats = perf_tracker.get_session_stats()
        logger.info(f"Stats after ending one session: {stats}")
        
        # Wait a bit to see if sessions are still active
        logger.info("Waiting 30 seconds...")
        time.sleep(30)
        
        # Check stats again
        stats = perf_tracker.get_session_stats()
        logger.info(f"Stats after 30 seconds: {stats}")
        
        # Change TTL to something even shorter for immediate expiry
        perf_tracker.set_session_ttl(0.01)  # ~36 seconds
        logger.info("Changed TTL to 0.01 hours (36 seconds)")
        
        # Start a new conversation
        new_session = perf_tracker.start_conversation(
            agent_id="demo_agent_new",
            user_id="user_new",
            metadata={"demo": True, "batch": 2}
        )
        if new_session:
            logger.info(f"Started new session: {new_session}")
        
        # Check final stats
        stats = perf_tracker.get_session_stats()
        logger.info(f"Final session stats: {stats}")
        
        # Try to end a session that might have expired
        if len(sessions) > 1:
            result = perf_tracker.end_conversation(
                session_id=sessions[1],
                quality_score=ConversationQuality.AVERAGE
            )
            logger.info(f"Attempted to end potentially expired session: {result}")
        
        # Show all session IDs and their format
        logger.info("Session ID formats:")
        for i, session_id in enumerate(sessions + ([new_session] if new_session else [])):
            if session_id:
                parts = session_id.split('_')
                logger.info(f"  Session {i+1}: {session_id}")
                if len(parts) >= 3:
                    agent_id = parts[0]
                    timestamp = parts[1]
                    random_part = parts[2]
                    logger.info(f"    Agent: {agent_id}, Timestamp: {timestamp}, Random: {random_part}")
        
        # Wait for cleanup cycle (demo purposes)
        logger.info("Waiting for cleanup cycle...")
        time.sleep(70)  # Wait for cleanup
        
        final_stats = perf_tracker.get_session_stats()
        logger.info(f"Stats after cleanup cycle: {final_stats}")
        
    except Exception as e:
        logger.error(f"Error during TTL demo: {e}")
    finally:
        perf_tracker.close()
        logger.info("TTL demo completed")

def session_format_demo():
    """Demonstrate session ID format and parsing"""
    logger.info("\n=== Session ID Format Demo ===")
    
    perf_tracker = AgentPerformanceTracker(
        base_url="https://your-backend-api.com",
        session_ttl_hours=24.0  # Normal TTL
    )
    
    try:
        # Create sessions with different agent IDs
        agents = ["hr_bot", "support_agent_1", "sales_ai"]
        sessions = []
        
        for agent_id in agents:
            session_id = perf_tracker.start_conversation(
                agent_id=agent_id,
                user_id=f"user_for_{agent_id}",
                metadata={"demo": "format_test"}
            )
            if session_id:
                sessions.append((agent_id, session_id))
                logger.info(f"Created session for {agent_id}: {session_id}")
        
        # Show format breakdown
        logger.info("\nSession ID Format Breakdown:")
        logger.info("Format: {agent_id}_{timestamp}_{random_8_chars}")
        
        for agent_id, session_id in sessions:
            parts = session_id.split('_')
            if len(parts) >= 3:
                timestamp = int(parts[1])
                import datetime
                dt = datetime.datetime.fromtimestamp(timestamp)
                logger.info(f"  {session_id}")
                logger.info(f"    Agent: {parts[0]}")
                logger.info(f"    Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"    Random: {parts[2]}")
        
        # Test session validation by ending conversations
        logger.info("\nTesting session validation:")
        for agent_id, session_id in sessions:
            result = perf_tracker.end_conversation(
                session_id=session_id,
                quality_score=ConversationQuality.EXCELLENT
            )
            logger.info(f"Ended {session_id}: {result}")
    
    except Exception as e:
        logger.error(f"Error during format demo: {e}")
    finally:
        perf_tracker.close()

if __name__ == "__main__":
    if not os.getenv('AGENT_TRACKER_API_KEY'):
        logger.warning("No API key set, but demo will still show local functionality")
    
    ttl_demo()
    session_format_demo() 