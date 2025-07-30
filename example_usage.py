#!/usr/bin/env python3
"""
AI Agent SDK Usage Examples - Hybrid Session Management

This script demonstrates the hybrid session management capabilities where the SDK
maintains lightweight local cache with backend persistence and seamless resumption.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from tracker import (
    AgentOperationsTracker, 
    AgentPerformanceTracker, 
    ConversationQuality,
    AgentStatus
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def hybrid_session_demo():
    """Demonstrate hybrid session management with backend fallback"""
    logger.info("=== Hybrid Session Management Demo ===")
    
    # Initialize with hybrid configuration
    perf_tracker = AgentPerformanceTracker(
        base_url="https://your-backend-api.com",
        api_key=os.getenv('AGENT_TRACKER_API_KEY'),
        session_ttl_hours=10.0,      # Local cache TTL (sliding)
        backend_ttl_hours=20.0,      # Backend persistence TTL  
        cleanup_interval_minutes=30,
        logger=logger
    )
    
    try:
        logger.info("Creating conversation sessions...")
        
        # Start multiple conversations
        session1 = perf_tracker.start_conversation(
            agent_id="support_agent_001",
            user_id="customer_123",
            metadata={"channel": "web_chat", "priority": "high"}
        )
        
        session2 = perf_tracker.start_conversation(
            agent_id="sales_agent_002", 
            user_id="lead_456",
            metadata={"channel": "phone", "campaign": "summer_sale"}
        )
        
        logger.info(f"Session 1: {session1}")
        logger.info(f"Session 2: {session2}")
        
        # Check initial session stats
        stats = perf_tracker.get_session_stats()
        logger.info(f"Initial session stats: {stats}")
        
        # Simulate active conversation (touches sessions)
        logger.info("\n--- Simulating Active Conversations ---")
        for i in range(3):
            time.sleep(2)
            
            # Check session status (automatically touches sessions)
            active1 = perf_tracker.is_session_active(session1)
            active2 = perf_tracker.is_session_active(session2)
            
            logger.info(f"Interval {i+1}: Session1 active: {active1}, Session2 active: {active2}")
            
            # Get TTL remaining
            ttl1 = perf_tracker.get_session_ttl_remaining(session1)
            ttl2 = perf_tracker.get_session_ttl_remaining(session2)
            
            logger.info(f"  TTL remaining - Session1: {ttl1:.3f}h, Session2: {ttl2:.3f}h")
        
        # Demonstrate session resumption after "crash"
        logger.info("\n--- Simulating SDK Restart/Crash ---")
        logger.info("Clearing local cache to simulate restart...")
        
        # Clear local cache to simulate crash/restart
        with perf_tracker._cache_lock:
            perf_tracker._session_cache.clear()
        
        logger.info("Local cache cleared. Now trying to access sessions...")
        
        # Try to access sessions - should retrieve from backend
        active1_after = perf_tracker.is_session_active(session1)
        active2_after = perf_tracker.is_session_active(session2)
        
        logger.info(f"After cache clear - Session1 active: {active1_after}, Session2 active: {active2_after}")
        
        if active1_after:
            logger.info("✅ Session1 successfully retrieved from backend!")
        if active2_after:
            logger.info("✅ Session2 successfully retrieved from backend!")
        
        # End conversations with different outcomes
        logger.info("\n--- Ending Conversations ---")
        
        # Successful conversation end
        if session1:
            success = perf_tracker.end_conversation(
                session_id=session1,
                quality_score=ConversationQuality.EXCELLENT,
                user_feedback="Very helpful support!",
                message_count=15,
                metadata={"resolution": "solved", "satisfaction": 5}
            )
            logger.info(f"Session1 ended successfully: {success}")
        
        # Failed conversation
        if session2:
            success = perf_tracker.record_failed_session(
                session_id=session2,
                error_message="Customer disconnected",
                metadata={"failure_reason": "timeout", "duration_before_failure": 300}
            )
            logger.info(f"Session2 recorded as failed: {success}")
        
        # Final session stats
        final_stats = perf_tracker.get_session_stats()
        logger.info(f"Final session stats: {final_stats}")
        
    except Exception as e:
        logger.error(f"Error in hybrid session demo: {e}")
    finally:
        perf_tracker.close()
        logger.info("Hybrid session demo completed")

def session_resumption_demo():
    """Demonstrate explicit session resumption with context"""
    logger.info("\n=== Session Resumption Demo ===")
    
    perf_tracker = AgentPerformanceTracker(
        base_url="https://your-backend-api.com",
        api_key=os.getenv('AGENT_TRACKER_API_KEY'),
        session_ttl_hours=0.01,      # Very short local TTL for demo
        backend_ttl_hours=20.0
    )
    
    try:
        # Start a conversation
        session_id = perf_tracker.start_conversation(
            agent_id="advisor_agent_003",
            user_id="client_789",
            metadata={"type": "financial_consultation"}
        )
        
        logger.info(f"Started consultation session: {session_id}")
        
        # Wait for local TTL to expire
        logger.info("Waiting for local TTL to expire...")
        time.sleep(60)  # Wait longer than local TTL
        
        # Try to access - should retrieve from backend
        logger.info("Checking session status after local TTL expiry...")
        still_active = perf_tracker.is_session_active(session_id)
        logger.info(f"Session still active: {still_active}")
        
        if still_active:
            logger.info("✅ Session seamlessly resumed from backend!")
            
            # Resume explicitly with context
            context = {
                "previous_topic": "investment_portfolio",
                "user_preferences": {"risk_tolerance": "moderate"},
                "conversation_stage": "recommendation_phase"
            }
            
            resumed = perf_tracker.resume_conversation(
                session_id=session_id,
                agent_id="advisor_agent_003",
                user_id="client_789",
                context=context,
                metadata={"resumption_reason": "session_recovery"}
            )
            
            logger.info(f"Explicit resumption with context: {resumed}")
            
            # Continue and end the conversation
            success = perf_tracker.end_conversation(
                session_id=session_id,
                quality_score=ConversationQuality.GOOD,
                user_feedback="Good advice, will consider the recommendations",
                message_count=28,
                metadata={"outcome": "recommendations_provided"}
            )
            logger.info(f"Resumed conversation ended: {success}")
        
    except Exception as e:
        logger.error(f"Error in resumption demo: {e}")
    finally:
        perf_tracker.close()

async def async_hybrid_demo():
    """Demonstrate async hybrid session management"""
    logger.info("\n=== Async Hybrid Session Demo ===")
    
    perf_tracker = AgentPerformanceTracker(
        base_url="https://your-backend-api.com",
        api_key=os.getenv('AGENT_TRACKER_API_KEY'),
        session_ttl_hours=10.0,
        backend_ttl_hours=20.0
    )
    
    try:
        # Start async conversation
        session_id = await perf_tracker.start_conversation_async(
            agent_id="chatbot_v2",
            user_id="web_user_001",
            metadata={"platform": "website", "entry_point": "help_button"}
        )
        
        logger.info(f"Async session started: {session_id}")
        
        # Simulate async operations
        await asyncio.sleep(1)
        
        # Check session with async fallback
        session_info = await perf_tracker._get_session_with_fallback_async(session_id)
        if session_info:
            logger.info(f"Async session retrieved: {session_info.agent_id}")
        
        # Resume with async
        if session_id:
            resumed = await perf_tracker.resume_conversation_async(
                session_id=session_id,
                agent_id="chatbot_v2",
                user_id="web_user_001",
                context={"interaction_count": 5},
                metadata={"async_resumption": True}
            )
            logger.info(f"Async resumption: {resumed}")
        
        # End async conversation
        if session_id:
            ended = await perf_tracker.end_conversation_async(
                session_id=session_id,
                quality_score=ConversationQuality.FAIR,
                message_count=12,
                metadata={"completion_type": "async"}
            )
            logger.info(f"Async conversation ended: {ended}")
        
    except Exception as e:
        logger.error(f"Error in async demo: {e}")
    finally:
        await perf_tracker.close_async()

def main():
    """Run all demos"""
    if not os.getenv('AGENT_TRACKER_API_KEY'):
        logger.warning("No API key set, but demo will still show local functionality")
    
    try:
        # Run sync demos
        hybrid_session_demo()
        session_resumption_demo()
        
        # Run async demo
        asyncio.run(async_hybrid_demo())
        
        logger.info("\n🎉 All hybrid session management demos completed!")
        logger.info("\nKey Benefits Demonstrated:")
        logger.info("- ✅ Lightweight local caching with sliding TTL")
        logger.info("- ✅ Automatic backend fallback and retrieval")
        logger.info("- ✅ Seamless session resumption after crashes")
        logger.info("- ✅ Context preservation across restarts")
        logger.info("- ✅ Dual TTL system (10h local + 20h backend)")
        logger.info("- ✅ Rich session analytics and monitoring")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    main() 