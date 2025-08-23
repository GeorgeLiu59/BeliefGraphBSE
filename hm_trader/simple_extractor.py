"""
Simple field extractor for LLM responses.
NO fallbacks - extract or raise error.
"""
import re
from typing import Dict, Any

def extract_price_prediction(text: str) -> Dict[str, Any]:
    """
    Extract predicted_other_trader_next_price from natural language.
    NO FALLBACKS - raises ValueError if not found.
    """
    # Look for price mentions in text - more aggressive patterns
    patterns = [
        r'around\s+(\d+)',
        r'price\s+of\s+(\d+)',
        r'at\s+(\d+)',
        r'about\s+(\d+)',
        r'approximately\s+(\d+)',
        r'roughly\s+(\d+)',
        r'near\s+(\d+)',
        r'quote.*?(\d+)',
        r'predict.*?(\d+)',
        r'(\d+)\s*-\s*\d+',  # For ranges like "150-155", take first
        r'between\s+(\d+)',
        r'(\d+)\s*cents?',  # "150 cents" or "150 cent"
        r'(\d+)\s*price',   # "150 price"
        r'next.*?(\d+)',    # "next 150"
        r'likely.*?(\d+)',  # "likely 150"
        r'(\d+)\s*range',   # "150 range"
        r'(\d+)\s*level',   # "150 level"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            price = int(match.group(1))
            if 1 <= price <= 500:  # Valid BSE price range
                # Extract confidence - more flexible patterns
                conf_patterns = [
                    r'(\d+(?:\.\d+)?)\s*%',           # "75%" or "0.75%"
                    r'confidence.*?(\d+\.\d+)',        # "confidence 0.75"
                    r'(\d+)\s*percent',                # "75 percent"
                    r'(\d+)\s*out\s*of\s*100',        # "75 out of 100"
                    r'(\d+)\s*per\s*cent',             # "75 per cent"
                    r'certainty.*?(\d+\.\d+)',         # "certainty 0.75"
                    r'(\d+)\s*confidence',              # "75 confidence"
                ]
                
                confidence = None
                for conf_pattern in conf_patterns:
                    conf_match = re.search(conf_pattern, text, re.IGNORECASE)
                    if conf_match:
                        val = float(conf_match.group(1))
                        # Convert percentage to decimal if needed
                        if val > 1:  # Likely a percentage
                            confidence = val / 100
                        else:  # Already decimal
                            confidence = val
                        break
                
                if confidence is None:
                    raise ValueError(f"Could not extract confidence from response: {text[:500]}")
                
                return {
                    "predicted_other_trader_next_price": price,
                    "confidence": confidence,
                    "reasoning": text[:200]  # First 200 chars as reasoning
                }
    
    # No price found - raise error (NO FALLBACK)
    raise ValueError(f"Could not extract price from response: {text[:500]}")

def extract_strategy(text: str) -> Dict[str, Any]:
    """
    Extract strategy description from natural language.
    NO FALLBACKS - raises ValueError if not found.
    """
    # Just use the whole response as strategy if it has strategy-related words
    strategy_words = ['aggressive', 'conservative', 'momentum', 'trend', 'pattern', 'strategy', 'behavior']
    
    if any(word in text.lower() for word in strategy_words):
        # Try to extract confidence
        conf_match = re.search(r'(\d+(?:\.\d+)?)\s*%|confidence.*?(\d+\.\d+)', text, re.IGNORECASE)
        if not conf_match:
            raise ValueError(f"Could not extract confidence from response: {text[:500]}")
        
        confidence = float(conf_match.group(1) or conf_match.group(2)) / 100 if conf_match.group(1) else float(conf_match.group(2))
        
        return {
            "possible_other_player_strategy": text[:300],  # First 300 chars
            "strategy_confidence": confidence
        }
    
    raise ValueError(f"Could not extract strategy from response: {text[:500]}")

def extract_my_quote(text: str) -> Dict[str, Any]:
    """
    Extract my_next_quote_price from natural language.
    NO FALLBACKS - raises ValueError if not found.  
    """
    # Similar to price prediction but for own quotes
    patterns = [
        r'quote.*?(\d+)',
        r'bid.*?(\d+)',
        r'ask.*?(\d+)',
        r'price.*?(\d+)',
        r'at\s+(\d+)',
        r'(\d+)\s*cents',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            price = int(match.group(1))
            if 1 <= price <= 500:
                return {
                    "my_next_quote_price": price,
                    "reasoning": text[:200]
                }
    
    raise ValueError(f"Could not extract quote price from response: {text[:500]}")
