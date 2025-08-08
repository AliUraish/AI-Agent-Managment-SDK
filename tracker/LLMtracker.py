#!/usr/bin/env python3
"""
LLM Usage Tracker

Tracks LLM model usage and token consumption, sending data to backend.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import from AgentOper for API client and config
from .AgentOper import SecureAPIClient, APIConfig

class LLMTracker:
    """Tracks LLM model usage and token consumption"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, client_id: Optional[str] = None):
        """Initialize LLM Tracker
        
        Args:
            base_url: API base URL
            api_key: API authentication key
            client_id: Client identifier
        """
        self.base_url = base_url.rstrip('/')
        self.api_client = SecureAPIClient(
            config=APIConfig(
                base_url=base_url,
                api_key=api_key,
                client_id=client_id
            )
        )
        self.logger = logging.getLogger(__name__)

    def _validate_provider(self, provider: str) -> str:
        """Validate and normalize provider name"""
        valid_providers = {"openai", "anthropic", "gemini"}
        provider_lower = provider.lower()
        if provider_lower not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}. Must be one of {valid_providers}")
        return provider_lower

    def _ensure_int(self, value: Any) -> int:
        """Ensure value is an integer"""
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to integer")
        return int(value)

    def _prepare_payload(self, provider: str, model: str, prompt_tokens: int, 
                        completion_tokens: int, session_id: Optional[str] = None,
                        agent_id: Optional[str] = None, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """Prepare the exact JSON payload expected by backend"""
        # Use current timestamp if not provided
        if not timestamp:
            timestamp = datetime.now().isoformat() + "Z"
        
        payload = {
            "timestamp": timestamp,
            "provider": self._validate_provider(provider),
            "model": model,
            # SDK original fields
            "prompt_tokens": self._ensure_int(prompt_tokens),
            "completion_tokens": self._ensure_int(completion_tokens),
            "total_tokens": self._ensure_int(prompt_tokens) + self._ensure_int(completion_tokens),
            # Backend-compatible aliases
            "tokens_input": self._ensure_int(prompt_tokens),
            "tokens_output": self._ensure_int(completion_tokens),
            "client_id": self.api_client.config.client_id
        }
        
        # Add optional fields
        if session_id:
            payload["session_id"] = session_id
        if agent_id:
            payload["agent_id"] = agent_id
            
        return payload

    def record_usage(self, provider: str, model: str, prompt_tokens: int, 
                    completion_tokens: int, session_id: Optional[str] = None,
                    agent_id: Optional[str] = None, timestamp: Optional[str] = None) -> bool:
        """Record LLM usage"""
        try:
            payload = self._prepare_payload(
                provider=provider,
                model=model, 
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                session_id=session_id,
                agent_id=agent_id,
                timestamp=timestamp
            )
            
            response = self.api_client.make_request('POST', '/api/sdk/llm-usage', payload)
            
            if response.success:
                self.logger.info(f"LLM usage recorded: {model} ({provider})")
                return True
            else:
                self.logger.error(f"Failed to record LLM usage: {response.error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error recording LLM usage: {e}")
            return False

    async def record_usage_async(self, provider: str, model: str, prompt_tokens: int, 
                                completion_tokens: int, session_id: Optional[str] = None,
                                agent_id: Optional[str] = None, timestamp: Optional[str] = None) -> bool:
        """Record LLM usage (async)"""
        try:
            payload = self._prepare_payload(
                provider=provider,
                model=model, 
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                session_id=session_id,
                agent_id=agent_id,
                timestamp=timestamp
            )
            
            response = await self.api_client.make_request_async('POST', '/api/sdk/llm-usage', payload)
            
            if response.success:
                self.logger.info(f"LLM usage recorded: {model} ({provider})")
                return True
            else:
                self.logger.error(f"Failed to record LLM usage: {response.error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error recording LLM usage: {e}")
            return False

    def close(self):
        """Close resources"""
        if hasattr(self.api_client, 'close'):
            self.api_client.close()

    async def close_async(self):
        """Close resources (async)"""
        if hasattr(self.api_client, 'close_async'):
            await self.api_client.close_async()