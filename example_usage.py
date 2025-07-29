#!/usr/bin/env python3
"""
Example usage of the separated Agent Operations and Performance Trackers
"""

import asyncio
import logging
import os
from tracker import (
    AgentOperationsTracker, 
    AgentPerformanceTracker,
    AgentStatus, 
    ConversationQuality
)

# Setup logging with appropriate level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get API key from environment variable (more secure than hardcoding)
API_KEY = os.getenv('AGENT_TRACKER_API_KEY')
if not API_KEY:
    logger.warning("No API key found in environment variables. Some features may be limited.")

def operations_example():
    """Example of Agent Operations Tracker usage"""
    logger.info("=== Agent Operations Tracker Example ===")
    
    # Initialize operations tracker
    ops_tracker = AgentOperationsTracker(
        base_url="https://your-backend-api.com",
        api_key=API_KEY,
        timeout=30,
        max_retries=3,
        logger=logger
    )
    
    try:
        # Register an agent
        success = ops_tracker.register_agent(
            agent_id="agent_001",
            sdk_version="1.0.0",
            metadata={
                "type": "customer_support",
                "region": "us-east",
                "environment": os.getenv('ENVIRONMENT', 'development')
            }
        )
        
        if success:
            logger.info("Agent registered successfully")
            
            # Update agent status
            ops_tracker.update_agent_status("agent_001", AgentStatus.ACTIVE)
            
            # Log some activities
            ops_tracker.log_activity(
                agent_id="agent_001",
                action="status_change",
                details={"from": "offline", "to": "active"}
            )
            
            ops_tracker.log_activity(
                agent_id="agent_001", 
                action="initialization_complete",
                details={"modules_loaded": 5, "startup_time": 2.3},
                duration=2.3
            )
        
        # Get active agents
        active_agents = ops_tracker.get_active_agents()
        if active_agents:
            logger.info("Retrieved active agents information")
        
        # Get recent activity
        activity = ops_tracker.get_recent_activity(limit=10, agent_id="agent_001")
        if activity:
            logger.info("Retrieved recent activity logs")
        
        # Get operations overview
        overview = ops_tracker.get_operations_overview()
        if overview:
            logger.info("Operations overview retrieved")
    
    except Exception as e:
        logger.error("Error during operations tracking: %s", str(e))
    finally:
        ops_tracker.close()

def performance_example():
    """Example of Agent Performance Tracker usage"""
    logger.info("=== Agent Performance Tracker Example ===")
    
    # Initialize performance tracker
    perf_tracker = AgentPerformanceTracker(
        base_url="https://your-backend-api.com",
        api_key=API_KEY,
        timeout=30,
        max_retries=3,
        logger=logger
    )
    
    try:
        # Start a conversation
        session_id = perf_tracker.start_conversation(
            agent_id="agent_001",
            user_id="user_123",
            metadata={
                "channel": "web",
                "priority": "high",
                "session_type": "support"
            }
        )
        
        if session_id:
            logger.info("Conversation started with ID: %s", session_id)
            
            # Simulate conversation end with quality score
            perf_tracker.end_conversation(
                session_id=session_id,
                quality_score=ConversationQuality.GOOD,
                user_feedback="Very helpful agent!",
                message_count=15,
                metadata={
                    "resolution": "solved",
                    "category": "technical"
                }
            )
            
            logger.info("Conversation completed successfully")
        
        # Start another conversation that fails
        failed_session = perf_tracker.start_conversation(
            agent_id="agent_001",
            user_id="user_456"
        )
        
        if failed_session:
            # Record a failed session
            perf_tracker.record_failed_session(
                session_id=failed_session,
                error_message="Connection timeout",
                metadata={
                    "error_code": "TIMEOUT_001",
                    "retry_count": 3
                }
            )
        
        # Get performance metrics
        success_rates = perf_tracker.get_success_rates(agent_id="agent_001")
        if success_rates:
            logger.info("Success rates retrieved")
        
        response_times = perf_tracker.get_response_times(agent_id="agent_001")
        if response_times:
            logger.info("Response times retrieved")
        
        quality_metrics = perf_tracker.get_conversation_quality(agent_id="agent_001")
        if quality_metrics:
            logger.info("Quality metrics retrieved")
        
        failed_sessions = perf_tracker.get_failed_sessions(agent_id="agent_001")
        if failed_sessions:
            logger.info("Failed sessions data retrieved")
        
        # Get performance overview
        perf_overview = perf_tracker.get_performance_overview(agent_id="agent_001")
        if perf_overview:
            logger.info("Performance overview retrieved")
        
        # Check session cache statistics
        session_stats = perf_tracker.get_session_stats()
        logger.info("Session cache stats: %s", session_stats)
    
    except Exception as e:
        logger.error("Error during performance tracking: %s", str(e))
    finally:
        perf_tracker.close()

async def async_example():
    """Example asynchronous usage of both trackers"""
    logger.info("=== Async Trackers Example ===")
    
    # Initialize both trackers with async support
    ops_tracker = AgentOperationsTracker(
        base_url="https://your-backend-api.com",
        api_key=API_KEY,
        enable_async=True,
        logger=logger
    )
    
    perf_tracker = AgentPerformanceTracker(
        base_url="https://your-backend-api.com",
        api_key=API_KEY,
        enable_async=True,
        logger=logger
    )
    
    try:
        # Operations tracking
        await ops_tracker.register_agent_async(
            agent_id="agent_002",
            sdk_version="1.0.0",
            metadata={"type": "chatbot", "environment": "production"}
        )
        
        await ops_tracker.update_agent_status_async("agent_002", AgentStatus.ACTIVE)
        
        await ops_tracker.log_activity_async(
            agent_id="agent_002",
            action="async_initialization",
            details={"async_mode": True, "startup_time": 1.8}
        )
        
        # Performance tracking
        session_id = await perf_tracker.start_conversation_async(
            agent_id="agent_002",
            user_id="user_789",
            metadata={"session_type": "automated"}
        )
        
        if session_id:
            # End conversation with performance data
            perf_tracker.end_conversation(
                session_id=session_id,
                quality_score=ConversationQuality.EXCELLENT,
                message_count=8
            )
        
        # Get async metrics
        active_agents = await ops_tracker.get_active_agents_async()
        success_rates = await perf_tracker.get_success_rates_async(agent_id="agent_002")
        
        if active_agents and success_rates:
            logger.info("Async operations and performance data retrieved")
    
    except Exception as e:
        logger.error("Error during async tracking: %s", str(e))
    finally:
        await ops_tracker.close_async()
        await perf_tracker.close_async()

def main():
    """Run all examples"""
    # Verify API key is available
    if not API_KEY:
        logger.warning("Please set AGENT_TRACKER_API_KEY environment variable")
    
    # Run synchronous examples
    operations_example()
    performance_example()
    
    # Run async example
    asyncio.run(async_example())

if __name__ == "__main__":
    main() 