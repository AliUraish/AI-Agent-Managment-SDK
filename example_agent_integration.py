from typing import Dict, Any, Optional
import time
from tracker import (
    AgentPerformanceTracker,
    AgentOperationsTracker,
    LLMTracker,
    SecurityWrapper,
    ComplianceWrapper
)

class TrackedAIAgent:
    """Example AI Agent with SDK tracking integration"""
    
    def __init__(self, agent_id: str, base_url: str = "http://localhost:8080/api"):
        # Initialize trackers
        self.operations_tracker = AgentOperationsTracker(base_url=base_url)
        self.performance_tracker = AgentPerformanceTracker(base_url=base_url)
        self.llm_tracker = LLMTracker(base_url=base_url)
        
        # Add security and compliance wrappers
        self.performance_tracker = SecurityWrapper(self.performance_tracker)
        self.performance_tracker = ComplianceWrapper(self.performance_tracker)
        
        # Register agent
        self.agent_id = agent_id
        success = self.operations_tracker.register_agent(
            agent_id=agent_id,
            sdk_version="1.0.0",
            metadata={"agent_type": "example"}
        )
        if not success:
            raise RuntimeError("Failed to register agent")
        
        # Set initial status
        self.operations_tracker.update_agent_status(agent_id, "active")
        
        # Track agent initialization
        self.operations_tracker.log_activity(
            agent_id=agent_id,
            activity_type="initialization",
            metadata={"status": "success"}
        )
    
    def process_message(self, user_message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process a user message with full tracking"""
        try:
            # Start conversation tracking
            session_id = self.performance_tracker.start_conversation(
                agent_id=self.agent_id,
                metadata=metadata
            )
            if not session_id:
                raise RuntimeError("Failed to start conversation tracking")
            
            # Log user message
            self.performance_tracker.log_user_message(
                session_id=session_id,
                content=user_message,
                metadata=metadata
            )
            
            # Track LLM usage (if using an LLM)
            start_time = time.time()
            
            # Your agent's processing logic here
            # For example:
            # response = your_llm_call(user_message)
            response = f"Echo: {user_message}"  # Replace with your agent's logic
            
            # Calculate response time
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Track LLM usage
            self.llm_tracker.record_usage(
                provider="openai",  # or your LLM provider
                model="gpt-4",     # or your model
                prompt_tokens=len(user_message.split()),  # example token count
                completion_tokens=len(response.split()),   # example token count
                session_id=session_id
            )
            
            # Log agent response
            self.performance_tracker.log_agent_message(
                session_id=session_id,
                content=response,
                response_time_ms=response_time_ms,
                tokens_used=len(response.split()),  # example token count
                metadata={
                    "model": "gpt-4",
                    "response_type": "text"
                }
            )
            
            # End conversation with success
            self.performance_tracker.end_conversation(
                session_id=session_id,
                quality_score=4.5,  # example score
                message_count=2     # user message + agent response
            )
            
            return response
            
        except Exception as e:
            # Log failure
            if session_id:
                self.performance_tracker.record_failed_session(
                    session_id=session_id,
                    error_message=str(e)
                )
            
            # Update agent status on error
            self.operations_tracker.update_agent_status(self.agent_id, "error")
            
            # Re-raise the exception
            raise
    
    async def process_message_async(self, user_message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Async version of message processing with tracking"""
        try:
            # Start conversation tracking
            session_id = await self.performance_tracker.start_conversation_async(
                agent_id=self.agent_id,
                metadata=metadata
            )
            if not session_id:
                raise RuntimeError("Failed to start conversation tracking")
            
            # Log user message
            await self.performance_tracker.log_user_message_async(
                session_id=session_id,
                content=user_message,
                metadata=metadata
            )
            
            # Track LLM usage (if using an LLM)
            start_time = time.time()
            
            # Your agent's async processing logic here
            # For example:
            # response = await your_async_llm_call(user_message)
            response = f"Echo: {user_message}"  # Replace with your agent's logic
            
            # Calculate response time
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Track LLM usage
            await self.llm_tracker.record_usage_async(
                provider="openai",  # or your LLM provider
                model="gpt-4",     # or your model
                prompt_tokens=len(user_message.split()),  # example token count
                completion_tokens=len(response.split()),   # example token count
                session_id=session_id
            )
            
            # Log agent response
            await self.performance_tracker.log_agent_message_async(
                session_id=session_id,
                content=response,
                response_time_ms=response_time_ms,
                tokens_used=len(response.split()),  # example token count
                metadata={
                    "model": "gpt-4",
                    "response_type": "text"
                }
            )
            
            # End conversation with success
            await self.performance_tracker.end_conversation_async(
                session_id=session_id,
                quality_score=4.5,  # example score
                message_count=2     # user message + agent response
            )
            
            return response
            
        except Exception as e:
            # Log failure
            if session_id:
                await self.performance_tracker.record_failed_session_async(
                    session_id=session_id,
                    error_message=str(e)
                )
            
            # Update agent status on error
            await self.operations_tracker.update_agent_status_async(self.agent_id, "error")
            
            # Re-raise the exception
            raise
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.operations_tracker.close()
        self.performance_tracker.close()
        self.llm_tracker.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        await self.operations_tracker.close_async()
        await self.performance_tracker.close_async()
        await self.llm_tracker.close_async()


# Example usage
if __name__ == "__main__":
    # Synchronous usage
    with TrackedAIAgent(agent_id="example_agent_1") as agent:
        response = agent.process_message(
            "Hello, how are you?",
            metadata={"user_id": "user123"}
        )
        print(f"Response: {response}")
    
    # Async usage
    import asyncio
    
    async def main():
        async with TrackedAIAgent(agent_id="example_agent_2") as agent:
            response = await agent.process_message_async(
                "Hello from async!",
                metadata={"user_id": "user456"}
            )
            print(f"Async Response: {response}")
    
    asyncio.run(main()) 