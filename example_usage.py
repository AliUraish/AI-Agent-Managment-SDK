#!/usr/bin/env python3
"""
Example usage of the API-based Agent Operations Tracker with secure API key handling
"""

import asyncio
import logging
import os
from tracker.AgentOper import AgentOperationsTracker, AgentStatus, ConversationQuality

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

def main():
    """Example synchronous usage with secure API key handling"""
    
    # Initialize tracker with API key from environment
    tracker = AgentOperationsTracker(
        base_url="https://your-backend-api.com",
        api_key=API_KEY,  # From environment variable
        timeout=30,
        max_retries=3,
        logger=logger
    )
    
    try:
        # Register an agent
        success = tracker.register_agent(
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
            tracker.update_agent_status("agent_001", AgentStatus.ACTIVE)
            
            # Start a conversation
            session_id = tracker.start_conversation(
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
                tracker.end_conversation(
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
        
        # Get system overview for dashboard
        overview = tracker.get_system_overview()
        if overview:
            # Log without sensitive data
            logger.info("System overview received")
            logger.debug("Active agents: %d", overview.get('active_agents', 0))
    
    except Exception as e:
        logger.error("Error during tracker operation: %s", str(e))
    finally:
        tracker.close()

async def async_main():
    """Example asynchronous usage with secure API key handling"""
    
    # Initialize tracker with async support
    tracker = AgentOperationsTracker(
        base_url="https://your-backend-api.com",
        api_key=API_KEY,  # From environment variable
        enable_async=True,
        logger=logger
    )
    
    try:
        # Use async methods
        success = await tracker.register_agent_async(
            agent_id="agent_002",
            sdk_version="1.0.0",
            metadata={
                "type": "chatbot",
                "environment": os.getenv('ENVIRONMENT', 'development')
            }
        )
        
        if success:
            await tracker.update_agent_status_async("agent_002", AgentStatus.ACTIVE)
            
            session_id = await tracker.start_conversation_async(
                agent_id="agent_002",
                user_id="user_456",
                metadata={"session_type": "automated"}
            )
            
            if session_id:
                # Record a failed session
                tracker.record_failed_session(
                    session_id=session_id,
                    error_message="Connection timeout",
                    metadata={
                        "error_code": "TIMEOUT_001",
                        "retry_count": 3
                    }
                )
        
        # Get system overview asynchronously
        overview = await tracker.get_system_overview_async()
        if overview:
            logger.info("Async system overview received")
    
    except Exception as e:
        logger.error("Error during async tracker operation: %s", str(e))
    finally:
        await tracker.close()

if __name__ == "__main__":
    # Verify API key is available
    if not API_KEY:
        logger.warning("Please set AGENT_TRACKER_API_KEY environment variable")
    
    print("=== Synchronous Example ===")
    main()
    
    print("\n=== Asynchronous Example ===")
    asyncio.run(async_main()) 