"""
LLM Controller Module for Hypothetical-Minds Trading Agent

Handles all LLM API interactions using the Gemini API.
Provides clean async interface matching Hypothetical-Minds pattern.
"""

import os
import asyncio
import re
from typing import List, Dict, Any, Optional


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
        self.max_tokens = kwargs.get('max_tokens', 1000)
    
    async def async_batch_prompt(self, system_message: str, user_messages: List[str]) -> List[List[str]]:
        """
        EQUIVALENT: async_batch_prompt interface from HM AsyncGPTController
        
        Args:
            system_message: System prompt/context
            user_messages: List of user messages to process
            
        Returns:
            List of responses, each wrapped in a list for HM compatibility
            
        Raises:
            RuntimeError: If API call fails
        """
        responses = []
        
        for user_msg in user_messages:
            try:
                # Combine system and user messages for Gemini
                full_prompt = f"{system_message}\n\n{user_msg}"
                
                # Import here to avoid circular imports
                import google.generativeai as genai
                
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens
                    )
                )
                responses.append([response.text])
                
            except Exception as e:
                raise RuntimeError(f"Gemini API call failed: {e}")
        
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
        self.max_tokens = 1000
        
        # Get API key
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        # Initialize controller
        self.controller = None
        self.gemini_model = None
        
        if self.api_key:
            self._initialize_controller()
    
    def _initialize_controller(self):
        """Initialize Gemini controller"""
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
    
    async def get_response(self, system_message: str, user_message: str) -> str:
        """
        Get single response from LLM
        
        Args:
            system_message: System prompt
            user_message: User message
            
        Returns:
            Response text
            
        Raises:
            RuntimeError: If controller unavailable or API call fails
        """
        if not self.is_available():
            raise RuntimeError(f"LLM controller not available for {self.trader_id}")
        
        try:
            responses = await self.controller.async_batch_prompt(system_message, [user_message])
            return responses[0][0] if responses and len(responses) > 0 and len(responses[0]) > 0 else ""
        
        except Exception as e:
            raise RuntimeError(f"LLM decision call failed for {self.trader_id}: {e}")
    
    def extract_json_dict(self, response: str) -> Dict[str, Any]:
        """
        Extract dictionary from LLM response using HM parsing logic
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Parsed dictionary
            
        Raises:
            ValueError: If parsing fails
        """
        try:
            # Find JSON markers
            start_marker = "```json\n"
            end_marker = "\n```"         
            start_pos = response.find(start_marker)
            
            if start_pos == -1:
                # Try alternative JSON markers
                start_marker = "```json"
                start_pos = response.find(start_marker)
                
                if start_pos != -1:
                    start_index = start_pos + len(start_marker)
                    # Find the JSON content
                    brace_start = response.find('{', start_index)
                    if brace_start != -1:
                        brace_count = 0
                        end_index = brace_start
                        for i, char in enumerate(response[brace_start:], brace_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_index = i + 1
                                    break
                        dict_str = response[brace_start:end_index].strip()
                    else:
                        raise ValueError("No JSON object found after ```json marker")
                else:
                    # Try to find raw JSON in response
                    brace_start = response.find('{')
                    if brace_start != -1:
                        brace_count = 0
                        end_index = brace_start
                        for i, char in enumerate(response[brace_start:], brace_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_index = i + 1
                                    break
                        dict_str = response[brace_start:end_index].strip()
                    else:
                        # Fallback for price extraction
                        if "predicted_opponent_next_price" in response or "my_quote_price" in response:
                            numbers = re.findall(r'\b\d{1,3}\b', response)  # Find 1-3 digit numbers
                            if numbers:
                                price = int(numbers[0])
                                if "predicted_opponent_next_price" in response:
                                    return {"predicted_opponent_next_price": price, "confidence": 0.5}
                                elif "my_quote_price" in response:
                                    return {"my_quote_price": price, "reasoning": "Extracted from non-JSON response", "aggressiveness": "moderate"}
                        raise ValueError("JSON dictionary markers not found in LLM response")
            else:
                start_index = start_pos + len(start_marker)
                end_index = response.find(end_marker, start_index)
                if end_index == -1:
                    end_index = response.find("```", start_index)
                    if end_index == -1:
                        raise ValueError("JSON dictionary end markers not found in LLM response")
                dict_str = response[start_index: end_index].strip()

            # Process each line, skipping comments
            lines = dict_str.split('\n')
            cleaned_lines = []
            for line in lines:
                comment_index = line.find('#')
                if comment_index != -1:
                    line = line[:comment_index].strip()
                if line:
                    cleaned_lines.append(line)

            # Reassemble the cleaned string
            cleaned_dict_str = ' '.join(cleaned_lines)
         
            # Convert the JSON string into a dictionary
            import json
            extracted_dict = json.loads(cleaned_dict_str)
            return extracted_dict
            
        except Exception as e:
            raise ValueError(f"LLM response parsing failed for {self.trader_id}: {e}. Raw response: {response[:500]}")
    
    def extract_price_from_response(self, response_text: str) -> tuple[Optional[int], str]:
        """
        Extract trading price from LLM response (fallback method)
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Tuple of (price, reasoning)
        """
        price = None
        reasoning = response_text
        
        # Try to find a number in the response
        numbers = re.findall(r'\d+', response_text)
        if numbers:
            try:
                price = int(numbers[0])  # Take the first number found
                reasoning = f"Found price {price} in response: {response_text}"
            except ValueError:
                raise ValueError(f"Could not parse price from response for {self.trader_id}: {response_text}")
        else:
            # Use a reasonable default price for empty markets
            price = 150  # Mid-range default
            reasoning = f"LLM couldn't decide possibly due to empty market, using default price {price}. LLM response: {response_text}"
        
        return price, reasoning


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

CRITICAL JSON FORMAT REQUIREMENT:
- You MUST ALWAYS respond in strict JSON format with ```json markers
- NO explanations, NO additional text outside the JSON block
- System will crash if you deviate from JSON format
- Always use the exact JSON structure requested

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

REMEMBER: ALWAYS respond in JSON format only. No explanations outside JSON.
"""