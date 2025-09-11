#!/usr/bin/env python3
"""
LLM-Designed Attributes System

This module allows LLM agents to design their own custom trading attributes
instead of just scoring predefined ones. This is the next level of innovation
where agents invent their own personality dimensions.
"""

import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

@dataclass
class CustomAttribute:
    """A custom attribute designed by the LLM"""
    name: str
    value: float  # 0.0 to 1.0
    description: str
    impact_on_trading: str
    min_value: float = 0.0
    max_value: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CustomAttributeSet:
    """Set of custom attributes designed by the LLM"""
    attributes: List[CustomAttribute]
    design_rationale: str
    market_context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'attributes': [attr.to_dict() for attr in self.attributes],
            'design_rationale': self.design_rationale,
            'market_context': self.market_context
        }
    
    def get_attribute(self, name: str) -> Optional[CustomAttribute]:
        """Get attribute by name"""
        for attr in self.attributes:
            if attr.name.lower() == name.lower():
                return attr
        return None
    
    def get_attribute_value(self, name: str) -> float:
        """Get attribute value by name, return 0.5 if not found"""
        attr = self.get_attribute(name)
        return attr.value if attr else 0.5

class LLMAttributeDesigner:
    """LLM-powered attribute designer that creates custom attributes"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        """Initialize the LLM attribute designer"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.design_history: List[Dict[str, Any]] = []
    
    def design_custom_attributes(
        self, 
        market_context: Dict[str, Any],
        design_mode: str = "initial"  # "initial" or "adaptation"
    ) -> CustomAttributeSet:
        """Design custom attributes based on market context"""
        
        if design_mode == "initial":
            prompt = self._create_initial_design_prompt(market_context)
        else:
            prompt = self._create_adaptation_design_prompt(market_context)
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_llm_response(response.text, market_context)
        except Exception as e:
            print(f"Error designing attributes: {e}")
            return self._create_fallback_attributes(market_context)
    
    def _create_initial_design_prompt(self, market_context: Dict[str, Any]) -> str:
        """Create prompt for initial attribute design"""
        return f"""You are a trading agent designing your own custom trading personality for a financial market.

CURRENT MARKET CONDITIONS:
- Market volatility: {market_context.get('volatility', 'Unknown')}
- Recent price trend: {market_context.get('trend', 'Unknown')}
- Competition level: {market_context.get('competition', 'Unknown')}
- Available liquidity: {market_context.get('liquidity', 'Unknown')}

YOUR MISSION:
Design 3-6 custom trading attributes that define your unique trading personality. These should be specific to your trading philosophy and the current market conditions.

EXAMPLES of custom attributes you might create:
- "trend_sensitivity" - How much you follow market trends
- "volatility_tolerance" - How much price volatility you can handle
- "patience_level" - How long you wait for optimal conditions
- "risk_appetite" - How much risk you're willing to take
- "market_timing" - How good you are at timing entries/exits
- "competition_awareness" - How much you consider other traders' actions

RESPOND WITH ONLY:
"ATTRIBUTES:
1. [attribute_name]: [value_0_to_1] - [description] - [impact_on_trading]
2. [attribute_name]: [value_0_to_1] - [description] - [impact_on_trading]
3. [attribute_name]: [value_0_to_1] - [description] - [impact_on_trading]
...

RATIONALE: [Brief explanation of why you chose these attributes and values]"

Example response:
"ATTRIBUTES:
1. trend_sensitivity: 0.8 - How much I follow market trends - Higher values mean I trade with momentum
2. volatility_tolerance: 0.6 - How much price swings I can handle - Higher values mean I accept more uncertainty
3. patience_level: 0.4 - How long I wait for good prices - Higher values mean I wait longer

RATIONALE: In this volatile market, I need to be trend-sensitive but not too patient, with moderate volatility tolerance."
"""
    
    def _create_adaptation_design_prompt(self, market_context: Dict[str, Any]) -> str:
        """Create prompt for attribute adaptation"""
        return f"""Your current trading attributes aren't working well. You need to redesign them.

CURRENT MARKET CONDITIONS:
- Market volatility: {market_context.get('volatility', 'Unknown')}
- Recent price trend: {market_context.get('trend', 'Unknown')}
- Competition level: {market_context.get('competition', 'Unknown')}
- Available liquidity: {market_context.get('liquidity', 'Unknown')}

PERFORMANCE ISSUES:
- Current profit: {market_context.get('profit', 0)}
- Market volatility: {market_context.get('market_volatility', 0)}
- Relative performance: {market_context.get('relative_performance', 0)}

YOUR MISSION:
Redesign your trading attributes to address your performance issues. You can:
1. Modify existing attributes
2. Add new attributes
3. Remove attributes that aren't working

RESPOND WITH ONLY:
"ATTRIBUTES:
1. [attribute_name]: [value_0_to_1] - [description] - [impact_on_trading]
2. [attribute_name]: [value_0_to_1] - [description] - [impact_on_trading]
...

RATIONALE: [Brief explanation of what you changed and why]"
"""
    
    def _parse_llm_response(self, response: str, market_context: Dict[str, Any]) -> CustomAttributeSet:
        """Parse LLM response into CustomAttributeSet"""
        try:
            lines = response.strip().split('\n')
            attributes = []
            rationale = ""
            
            in_attributes = False
            in_rationale = False
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("ATTRIBUTES:"):
                    in_attributes = True
                    continue
                elif line.startswith("RATIONALE:"):
                    in_attributes = False
                    in_rationale = True
                    rationale = line.replace("RATIONALE:", "").strip()
                    continue
                
                if in_attributes and line and line[0].isdigit():
                    # Parse attribute line: "1. name: value - description - impact"
                    try:
                        # Remove number prefix
                        attr_line = line.split('.', 1)[1].strip()
                        
                        # Split by colons and dashes
                        parts = attr_line.split(' - ')
                        if len(parts) >= 3:
                            name_value = parts[0].strip()
                            description = parts[1].strip()
                            impact = parts[2].strip()
                            
                            # Extract name and value
                            if ':' in name_value:
                                name, value_str = name_value.split(':', 1)
                                name = name.strip()
                                value = float(value_str.strip())
                                
                                # Validate value
                                value = max(0.0, min(1.0, value))
                                
                                attributes.append(CustomAttribute(
                                    name=name,
                                    value=value,
                                    description=description,
                                    impact_on_trading=impact
                                ))
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing attribute line: {line} - {e}")
                        continue
            
            # If no attributes were parsed, create fallback
            if not attributes:
                return self._create_fallback_attributes(market_context)
            
            # Record design
            self.design_history.append({
                'timestamp': len(self.design_history),
                'market_context': market_context,
                'attributes': [attr.to_dict() for attr in attributes],
                'rationale': rationale
            })
            
            return CustomAttributeSet(
                attributes=attributes,
                design_rationale=rationale,
                market_context=market_context
            )
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return self._create_fallback_attributes(market_context)
    
    def _create_fallback_attributes(self, market_context: Dict[str, Any]) -> CustomAttributeSet:
        """Create fallback attributes if LLM fails"""
        fallback_attributes = [
            CustomAttribute(
                name="trend_sensitivity",
                value=0.5,
                description="How much I follow market trends",
                impact_on_trading="Higher values mean I trade with momentum"
            ),
            CustomAttribute(
                name="volatility_tolerance",
                value=0.5,
                description="How much price swings I can handle",
                impact_on_trading="Higher values mean I accept more uncertainty"
            ),
            CustomAttribute(
                name="patience_level",
                value=0.5,
                description="How long I wait for good prices",
                impact_on_trading="Higher values mean I wait longer"
            )
        ]
        
        return CustomAttributeSet(
            attributes=fallback_attributes,
            design_rationale="Fallback attributes due to LLM parsing error",
            market_context=market_context
        )
    
    def get_design_history(self) -> List[Dict[str, Any]]:
        """Get history of attribute designs"""
        return self.design_history

class CustomAttributeManager:
    """Manager for custom LLM-designed attributes"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.attribute_set: Optional[CustomAttributeSet] = None
        self.designer = LLMAttributeDesigner()
        self.adaptation_enabled: bool = True
    
    def initialize_custom_attributes(self, market_context: Dict[str, Any]) -> CustomAttributeSet:
        """Initialize custom attributes using LLM"""
        self.attribute_set = self.designer.design_custom_attributes(market_context, "initial")
        return self.attribute_set
    
    def adapt_custom_attributes(self, market_context: Dict[str, Any]) -> Optional[CustomAttributeSet]:
        """Adapt custom attributes based on performance"""
        if not self.adaptation_enabled or self.attribute_set is None:
            return None
        
        # Check if adaptation is needed
        if self._should_adapt(market_context):
            new_attributes = self.designer.design_custom_attributes(market_context, "adaptation")
            self.attribute_set = new_attributes
            return new_attributes
        
        return None
    
    def _should_adapt(self, market_context: Dict[str, Any]) -> bool:
        """Determine if attributes should be adapted"""
        profit = market_context.get('profit', 0)
        volatility = market_context.get('market_volatility', 0)
        relative_performance = market_context.get('relative_performance', 0)
        
        # Adapt if performance is poor
        if profit < -50 or relative_performance < -0.2 or volatility > 0.8:
            return True
        
        return False
    
    def get_attributes(self) -> CustomAttributeSet:
        """Get current custom attributes"""
        if self.attribute_set is None:
            raise ValueError("Custom attributes not initialized. Call initialize_custom_attributes() first.")
        return self.attribute_set
    
    def to_json(self) -> str:
        """Convert to JSON"""
        data = {
            'agent_id': self.agent_id,
            'attribute_set': self.attribute_set.to_dict() if self.attribute_set else None,
            'adaptation_enabled': self.adaptation_enabled,
            'design_history': self.designer.get_design_history()
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CustomAttributeManager':
        """Create from JSON"""
        data = json.loads(json_str)
        manager = cls(data['agent_id'])
        manager.adaptation_enabled = data['adaptation_enabled']
        
        if data['attribute_set']:
            attr_data = data['attribute_set']
            attributes = [CustomAttribute(**attr) for attr in attr_data['attributes']]
            manager.attribute_set = CustomAttributeSet(
                attributes=attributes,
                design_rationale=attr_data['design_rationale'],
                market_context=attr_data['market_context']
            )
        
        return manager

# Example usage and testing
def test_custom_attribute_design():
    """Test the custom attribute design system"""
    print("Testing LLM-Designed Custom Attributes")
    print("=" * 60)
    
    # Test different market conditions
    market_conditions = [
        {
            'volatility': 'Low',
            'trend': 'Sideways',
            'competition': 'Low',
            'liquidity': 'High'
        },
        {
            'volatility': 'High',
            'trend': 'Upward',
            'competition': 'Moderate',
            'liquidity': 'High'
        },
        {
            'volatility': 'Extreme',
            'trend': 'Strong Upward',
            'competition': 'Intense',
            'liquidity': 'Very High'
        }
    ]
    
    for i, context in enumerate(market_conditions, 1):
        print(f"\nMARKET CONDITION {i}: {context['volatility']} volatility, {context['trend']} trend")
        print("-" * 50)
        
        try:
            manager = CustomAttributeManager(f"TEST_AGENT_{i}")
            attributes = manager.initialize_custom_attributes(context)
            
            print(f"Design Rationale: {attributes.design_rationale}")
            print(f"Custom Attributes:")
            
            for attr in attributes.attributes:
                print(f"  • {attr.name}: {attr.value:.2f}")
                print(f"    Description: {attr.description}")
                print(f"    Impact: {attr.impact_on_trading}")
                print()
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_custom_attribute_design()