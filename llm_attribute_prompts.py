#!/usr/bin/env python3
"""
LLM Prompt Templates for Agent Attribute Design

This module contains the prompt templates that LLM agents use to:
1. Design their initial trading attributes
2. Adapt their attributes based on performance
3. Reason about their attribute choices
"""

from typing import Dict, Any, List, Optional


class AttributePromptTemplates:
    """Templates for LLM prompts related to attribute design and adaptation"""
    
    @staticmethod
    def create_initial_design_prompt(
        market_context: Dict[str, Any],
        available_strategies: List[str]
    ) -> str:
        """Create prompt for initial attribute design"""
        
        strategies_text = "\n".join([f"- {s}" for s in available_strategies])
        
        prompt = f"""You are a trading agent designing your own trading personality for a financial market.

CURRENT MARKET CONDITIONS:
- Market volatility: {market_context.get('volatility', 'Unknown')}
- Recent price trend: {market_context.get('trend', 'Unknown')}
- Competition level: {market_context.get('competition', 'Unknown')}
- Available liquidity: {market_context.get('liquidity', 'Unknown')}

AVAILABLE DESIGN STRATEGIES:
{strategies_text}

YOUR MISSION:
Design your trading personality by choosing one of the available strategies. Each strategy creates a different combination of trading attributes that will define how you behave in the market.

RESPOND WITH ONLY:
"DESIGN: [strategy_name]"

No explanation needed. Choose the strategy that best fits your understanding of the current market conditions and your trading philosophy.

Examples:
- "DESIGN: conservative"
- "DESIGN: aggressive"
- "DESIGN: balanced"
- "DESIGN: momentum"
- "DESIGN: mean_reversion"
- "DESIGN: random"
"""
        return prompt
    
    @staticmethod
    def create_adaptation_prompt(
        current_attributes: Dict[str, float],
        performance_metrics: Dict[str, Any],
        market_context: Dict[str, Any],
        available_strategies: List[str]
    ) -> str:
        """Create prompt for attribute adaptation"""
        
        strategies_text = "\n".join([f"- {s}" for s in available_strategies])
        
        # Format current attributes for display
        attr_display = "\n".join([
            f"- {attr.replace('_', ' ').title()}: {value:.2f}"
            for attr, value in current_attributes.items()
        ])
        
        prompt = f"""Your current trading attributes aren't working well in this market. You need to adapt.

CURRENT ATTRIBUTES:
{attr_display}

PERFORMANCE METRICS:
- Current profit: ${performance_metrics.get('profit', 0)}
- Market volatility: {performance_metrics.get('market_volatility', 0):.2f}
- Relative performance: {performance_metrics.get('relative_performance', 0):.2f}
- Number of trades: {performance_metrics.get('trade_count', 0)}

MARKET CONDITIONS:
- Current volatility: {market_context.get('volatility', 'Unknown')}
- Price trend: {market_context.get('trend', 'Unknown')}
- Competition: {market_context.get('competition', 'Unknown')}

AVAILABLE ADAPTATION STRATEGIES:
{strategies_text}

ANALYZE what went wrong and ADAPT your attributes by choosing a new strategy.

RESPOND WITH ONLY:
"ADAPT: [strategy_name]"

REASONING: [Brief explanation of why you're changing - max 50 words]

Examples:
- "ADAPT: conservative"
- "ADAPT: aggressive"
- "ADAPT: balanced"

Choose the strategy that addresses your current performance issues.
"""
        return prompt
    
    @staticmethod
    def create_attribute_influenced_trading_prompt(
        attributes: Dict[str, float],
        market_context: Dict[str, Any],
        belief_graph_insights: Dict[str, Any]
    ) -> str:
        """Create trading prompt that incorporates the agent's attributes"""
        
        # Create personality description based on attributes
        personality = AttributePromptTemplates._describe_personality(attributes)
        
        prompt = f"""You are a trading agent with a specific personality that influences your decisions.

YOUR TRADING PERSONALITY:
{personality}

MARKET CONTEXT:
- Best bid: {market_context.get('best_bid', 'None')}
- Best ask: {market_context.get('best_ask', 'None')}
- Recent trades: {market_context.get('recent_trades', [])}
- Market volatility: {market_context.get('volatility', 'Unknown')}

BELIEF GRAPH INSIGHTS:
- Other agents' strategies: {belief_graph_insights.get('agent_strategies', 'None')}
- Market sentiment: {belief_graph_insights.get('market_sentiment', 'None')}
- Risk assessment: {belief_graph_insights.get('risk_assessment', 'None')}

YOUR MISSION:
Make a trading decision that aligns with your personality and the current market conditions.

RESPOND WITH ONLY:
"BUY [price]" - to place a bid at that price
"SELL [price]" - to place an ask at that price  
"WAIT" - to wait for better conditions

Your personality should guide your decision:
- High aggressiveness = more likely to trade at current prices
- High patience = more likely to wait for better prices
- High risk tolerance = more likely to accept price uncertainty
- High momentum following = more likely to follow trends
- High mean reversion = more likely to bet on price reversals
"""
        return prompt
    
    @staticmethod
    def _describe_personality(attributes: Dict[str, float]) -> str:
        """Create a natural language description of the agent's personality"""
        descriptions = []
        
        if attributes.get('aggressiveness', 0) > 0.7:
            descriptions.append("You are AGGRESSIVE - you pursue trading opportunities actively")
        elif attributes.get('aggressiveness', 0) < 0.3:
            descriptions.append("You are CONSERVATIVE - you wait for very good opportunities")
        else:
            descriptions.append("You are MODERATELY aggressive in pursuing trades")
        
        if attributes.get('patience', 0) > 0.7:
            descriptions.append("You are PATIENT - you wait for optimal conditions")
        elif attributes.get('patience', 0) < 0.3:
            descriptions.append("You are IMPATIENT - you trade frequently")
        else:
            descriptions.append("You have MODERATE patience for trading")
        
        if attributes.get('risk_tolerance', 0) > 0.7:
            descriptions.append("You have HIGH RISK TOLERANCE - you accept price uncertainty")
        elif attributes.get('risk_tolerance', 0) < 0.3:
            descriptions.append("You have LOW RISK TOLERANCE - you prefer safer trades")
        else:
            descriptions.append("You have MODERATE risk tolerance")
        
        if attributes.get('momentum_following', 0) > 0.7:
            descriptions.append("You FOLLOW MARKET MOMENTUM - you trade with trends")
        elif attributes.get('mean_reversion', 0) > 0.7:
            descriptions.append("You BET ON MEAN REVERSION - you expect prices to return to average")
        else:
            descriptions.append("You have BALANCED trend/momentum approach")
        
        if attributes.get('adaptability', 0) > 0.7:
            descriptions.append("You are HIGHLY ADAPTABLE - you change strategies quickly")
        elif attributes.get('adaptability', 0) < 0.3:
            descriptions.append("You are CONSISTENT - you stick to your initial strategy")
        else:
            descriptions.append("You are MODERATELY adaptable to market changes")
        
        return "\n".join(descriptions)


class AttributePromptParser:
    """Parser for LLM responses to attribute prompts"""
    
    @staticmethod
    def parse_design_response(response: str) -> str:
        """Parse the design strategy from LLM response"""
        response_upper = response.upper().strip()
        
        # Look for "DESIGN: [strategy]" pattern
        if "DESIGN:" in response_upper:
            strategy = response_upper.split("DESIGN:")[1].strip()
            return strategy.lower()
        
        # Fallback: look for strategy names in the response
        strategies = ["random", "conservative", "aggressive", "balanced", "momentum", "mean_reversion"]
        for strategy in strategies:
            if strategy.upper() in response_upper:
                return strategy
        
        # Default to balanced if no clear strategy found
        return "balanced"
    
    @staticmethod
    def parse_adaptation_response(response: str) -> tuple[str, str]:
        """Parse the adaptation strategy and reasoning from LLM response"""
        response_upper = response.upper().strip()
        
        # Look for "ADAPT: [strategy]" pattern
        strategy = "balanced"  # default
        if "ADAPT:" in response_upper:
            strategy_part = response_upper.split("ADAPT:")[1].split("\n")[0].strip()
            strategies = ["random", "conservative", "aggressive", "balanced", "momentum", "mean_reversion"]
            for s in strategies:
                if s.upper() in strategy_part:
                    strategy = s
                    break
        
        # Extract reasoning
        reasoning = "No reasoning provided"
        if "REASONING:" in response:
            reasoning_part = response.split("REASONING:")[1].strip()
            reasoning = reasoning_part[:100]  # Limit to 100 characters
        
        return strategy, reasoning
    
    @staticmethod
    def parse_trading_response(response: str) -> tuple[str, Optional[float]]:
        """Parse the trading decision from LLM response"""
        response_upper = response.upper().strip()
        
        # Look for BUY/SELL with price
        if "BUY" in response_upper:
            try:
                price_str = response_upper.split("BUY")[1].strip()
                price = float(price_str)
                return "BUY", price
            except (ValueError, IndexError):
                return "WAIT", None
        
        elif "SELL" in response_upper:
            try:
                price_str = response_upper.split("SELL")[1].strip()
                price = float(price_str)
                return "SELL", price
            except (ValueError, IndexError):
                return "WAIT", None
        
        # Default to wait
        return "WAIT", None
