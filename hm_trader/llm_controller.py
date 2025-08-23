"""
LLM Controller Module for Hypothetical-Minds Trading Agent

Handles all LLM API interactions using the Gemini API.
Provides clean async interface matching Hypothetical-Minds pattern.
"""

import os
import asyncio
import re
from typing import List, Dict, Any, Optional
# Unified extraction - use only JSON parsing, no regex fallbacks


class GeminiController:
    """
    LLM controller for Gemini API interactions.
    Equivalent to Hypothetical-Minds AsyncGPTController.
    """
    
    def __init__(self, model, model_id: str, **kwargs):
        """
        Initialize Gemini controller
        
        Args:
            model: Gemini model instance
            model_id: Model identifier string
            **kwargs: Additional configuration (temperature, max_tokens)
        """
        self.model = model
        self.model_id = model_id
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 32000)  # Increased for full reasoning text
    
    async def async_batch_prompt(self, expertise: str, messages: List[str], temperature: float = None) -> List[Dict[str, Any]]:
        """
        Simple extraction approach - NO schema, NO fallbacks
        """
        if temperature is None:
            temperature = self.temperature
            
        responses = []
        
        for user_msg in messages:
            # Get raw text response from LLM
            import google.generativeai as genai
            
            full_prompt = f"{expertise}\n\n{user_msg}"
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            if not response or not response.text:
                raise RuntimeError("Empty response from API")
            
            # Return raw response - let BSE agent's extract_dict handle parsing
            # This removes the confusing dual extraction system
            responses.append(response.text)
        
        return responses
    



class LLMInterface:
    """
    High-level interface for LLM operations in trading context.
    Handles initialization, configuration, and response parsing.
    """
    
    def __init__(self, trader_id: str, api_key: Optional[str] = None):
        """
        Initialize LLM interface
        
        Args:
            trader_id: Trader identifier for error reporting
            api_key: Google API key (uses env var if None)
        """
        self.trader_id = trader_id
        self.model_name = 'gemini-2.5-flash-lite'
        self.temperature = 0.7
        self.max_tokens = 32000  # Increased for full reasoning text
        
        # Get API key
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        # Initialize controller
        self.controller = None
        self.gemini_model = None
        
        if self.api_key:
            self._initialize_controller()
    
    def _initialize_controller(self):
        """
        (2) ENV SETUP: Initialize LLM connection and structured output capabilities
        
        INTUITION: This is where the agent gets its "brain" connected. We set up the
        communication channel to the LLM (Gemini) and configure it for structured
        reasoning. The agent can now think, reason, and make hypothesis-driven decisions.
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Create Gemini model instance
            self.gemini_model = genai.GenerativeModel(self.model_name)
            
            # Create controller
            model_args = {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
            
            self.controller = GeminiController(
                model=self.gemini_model,
                model_id=self.model_name,
                **model_args
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM controller for {self.trader_id}: {e}")
    
    def is_available(self) -> bool:
        """Check if LLM controller is available"""
        return self.controller is not None
    
    async def get_response(self, system_message: str, user_message: str) -> Dict[str, Any]:
        """
        Get single structured response from LLM
        
        Args:
            system_message: System prompt
            user_message: User message
            
        Returns:
            Structured response dictionary
            
        Raises:
            RuntimeError: If controller unavailable or API call fails
        """
        if not self.is_available():
            raise RuntimeError(f"LLM controller not available for {self.trader_id}")
        
        try:
            responses = await self.controller.async_batch_prompt(system_message, [user_message])
            return responses[0] if responses and len(responses) > 0 else {"response": "", "confidence": 0.0}
        
        except Exception as e:
            raise RuntimeError(f"LLM decision call failed for {self.trader_id}: {e}")
    
    async def async_batch_prompt(self, system_message: str, messages: List[str], temperature: float = None) -> List[Dict[str, Any]]:
        """
        Delegate to controller's async_batch_prompt method for HM compatibility
        
        Args:
            system_message: System message/expertise
            messages: List of user messages
            temperature: Optional temperature override
            
        Returns:
            List of structured response dictionaries
        """
        if not self.is_available():
            raise RuntimeError(f"LLM controller not available for {self.trader_id}")
            
        return await self.controller.async_batch_prompt(system_message, messages, temperature)


def create_system_message(trader_id: str) -> str:
    """
    Generate system message for trading LLM
    
    Args:
        trader_id: Trader identifier
        
    Returns:
        System message string
    """
    return f"""
You are Trader {trader_id} in the Bristol Stock Exchange (BSE) simulation.

RESPONSE FORMAT:
- Follow the EXACT format requested in each prompt
- If asked for a DECISION, provide the ANSWER in the specified format
- If asked for a PREDICT, provide the ANSWER in the specified format  
- If asked for a STRATEGY, provide the ANSWER in the specified format
- Keep responses short and direct
- Do NOT provide market analysis unless specifically requested

TRADING ENVIRONMENT:
- You are participating in a limit order book (LOB) financial exchange
- You can submit BID orders (to buy) or ASK orders (to sell) 
- Only one order per trader is allowed at any time (new order replaces old)
- Zero latency communications - all traders see LOB updates immediately
- Price range: 1 to 500 cents

MARKET MECHANICS:
- Limit Order Book (LOB) maintains best bid/ask prices
- Orders are matched when bid >= ask price
- You receive customer orders that specify limit prices and quantities
- Your goal is to maximize profit while fulfilling customer orders

OPPONENT MODELING (Hypothetical-Minds approach):
- Track competitor behavior patterns
- Form hypotheses about opponent strategies
- Update beliefs based on market interactions
- Use self-improvement to refine strategies over time

TRADING STRATEGY:
- Analyze current market conditions (best bid/ask, spread, recent trades)
- Consider your position (current orders, balance, recent performance)
- Model opponent behaviors and adapt accordingly
- Make informed decisions about quote prices to maximize profit
- Balance between aggressive pricing (more trades) vs conservative pricing (better margins)

DECISION MAKING:
- Always provide a specific quote price as an integer between 1-500
- Consider market momentum and competitor actions
- Adapt strategy based on hypothesis evaluation
- Use feedback to improve future decisions

IMPORTANT: Follow the prompt format exactly. Do not add explanations or analysis unless specifically asked.
"""