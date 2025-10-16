#!/usr/bin/env python3
"""
Unified Prompt Configuration for All Trading Agents

All LLM trading agents import their prompts from this single source.
Prompts are organized by:
1. Core system prompts (scaffolding-independent)
2. JSON belief graph scaffolding
3. Natural language belief graph scaffolding
4. Trading action prompts
5. CoT reasoning prompts

NO FALLBACKS. NO DEFENSIVE CODE. DIRECT EXECUTION.
"""

from typing import Dict, Any, Optional, List


class BasePrompts:
    """Core prompts shared across all agents"""

    MARKET_FUNDAMENTALS = """
MARKET EDUCATION:
- Price ranges typically between $1-$500 in this market
- The market operates as a continuous double auction with a limit order book
- Orders are matched when bid prices meet or exceed ask prices
- You can place BID orders (to buy) or ASK orders (to sell)

HOW ORDER BOOKS WORK:
- To BUY: Place a BID order at your desired price
  - If sellers exist at/below your bid price → immediate execution
  - If no sellers at your price → your bid waits on the order book
  - Higher bids are more likely to execute quickly

- To SELL: Place an ASK order at your desired price
  - If buyers exist at/above your ask price → immediate execution
  - If no buyers at your price → your ask waits on the order book
  - Lower asks are more likely to execute quickly

TRADING PRINCIPLES:
- Active trading generates more opportunities than passive waiting
- Velocity matters: completing trades quickly lets you capture new opportunities
- Strategic losses: taking a small loss now can free capital for bigger gains later
- Opportunity cost: holding inventory waiting for perfect prices means missing other trades
- Risk management: avoid spending your entire balance on one trade
- Trade frequently and learn from market dynamics
- Analyze other traders' activity patterns and adapt your strategy
"""

    @staticmethod
    def format_market_context(lob: Dict, time: float, trader_state: Dict) -> str:
        """Format market data consistently across all agents"""
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        bid_ask_spread = (best_ask - best_bid) if (best_bid and best_ask) else None

        recent_prices = trader_state.get('recent_prices', [])
        avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else None
        avg_price_str = f"${avg_price:.1f}" if avg_price else "N/A"

        avg_purchase = trader_state.get('avg_purchase_price')
        avg_purchase_str = f"${avg_purchase:.2f}" if avg_purchase else "None"

        last_purchase = trader_state.get('last_purchase_price')
        last_purchase_str = f"${last_purchase:.2f}" if last_purchase else "None"

        return f"""MARKET DATA:
Time: {time:.1f}
Best Bid: {best_bid}
Best Ask: {best_ask}
Spread: {bid_ask_spread}
Recent prices: {recent_prices}
Average recent price: {avg_price_str}

MY CURRENT STATE:
Balance: ${trader_state.get('balance', 0)}
Inventory: {trader_state.get('inventory', 0)} units
Average purchase price: {avg_purchase_str}
Last purchase price: {last_purchase_str}
Number of completed trades: {trader_state.get('n_trades', 0)}
"""


class BeliefGraphScaffolding:
    """Belief graph integration prompts - JSON vs Natural Language"""

    @staticmethod
    def json_format() -> str:
        """JSON belief graph scaffolding"""
        return """
BELIEF GRAPH INSIGHTS (JSON Format):
The belief graph tracks your understanding of market dynamics and other agents' strategies.

Current belief graph state:
{belief_graph_json}

Key elements:
- "strategy_beliefs": What strategies you believe other agents are following
- "market_sentiment": Overall market direction and momentum
- "risk_assessment": Current market risk level
- "confidence_scores": How confident you are in your beliefs about each agent

Use this structured data to inform your trading decisions.
"""

    @staticmethod
    def natural_language_format() -> str:
        """Natural language belief graph scaffolding"""
        return """
BELIEF GRAPH INSIGHTS (Natural Language):
The belief graph represents your understanding of market dynamics and other agents' strategies.

Current market beliefs:
{belief_graph_narrative}

This narrative describes:
- How aggressive or passive other traders are behaving
- What strategies you believe they are following
- Overall market sentiment and direction
- Your assessment of current market risks
- How confident you are in your understanding of each agent

Use these insights to inform your trading decisions.
"""


class ChainOfThoughtPrompts:
    """Chain-of-thought reasoning scaffolding"""

    @staticmethod
    def cot_reasoning_prefix() -> str:
        """Prefix for CoT-enabled agents"""
        return """
REASONING PROCESS:
Before making your decision, think through:
1. What is the current market situation and momentum?
2. What are other traders doing and what opportunities does that create?
3. What does my belief graph tell me about the market?
4. What's the opportunity cost of waiting vs acting now?
5. Should I take action now or wait? (Bias toward action when reasonable)

Provide your reasoning step-by-step, then state your decision.
"""

    @staticmethod
    def no_cot_suffix() -> str:
        """Instructions for non-CoT agents"""
        return """
Respond ONLY with your decision. No explanation needed.
"""


class GraphQualityVariants:
    """Different graph quality scenarios"""

    @staticmethod
    def perfect_graph_context() -> str:
        """Perfect graph with complete information"""
        return """
GRAPH QUALITY: PERFECT
You have complete and accurate information about all market participants and their strategies.
All belief graph data is verified and highly reliable.
"""

    @staticmethod
    def basic_graph_context() -> str:
        """Basic graph with partial information"""
        return """
GRAPH QUALITY: BASIC
You have limited information about market participants.
Belief graph data is based on observations and may contain uncertainties.
"""

    @staticmethod
    def no_graph_context() -> str:
        """No belief graph - baseline LLM"""
        return """
GRAPH QUALITY: NONE
You do not have access to belief graph insights.
Make decisions based solely on observable market data.
"""


class HypotheticalMindPrompts:
    """Hypothetical-Minds specific prompts"""

    @staticmethod
    def hm_system_message(agent_id: str) -> str:
        """System message for HM agents"""
        return f"""
You are Agent {agent_id} in the Bristol Stock Exchange (BSE) trading simulation.
This is a multi-agent financial trading environment where you compete with other traders
to maximize profit through strategic order placement.

Your goal is to maximize trading profit over time.

You use Hypothetical-Minds reasoning:
- Generate hypotheses about opponent strategies
- Evaluate hypotheses using Rescorla-Wagner learning
- Update beliefs based on prediction accuracy
- Adapt your strategy to exploit opponent weaknesses

Key strategic considerations:
- Track competitor trading patterns and adapt accordingly
- Balance aggressive vs conservative pricing strategies
- Consider market momentum and liquidity conditions
- Model opponent strategies to predict their future actions
- Use hypothesis-driven reasoning to improve decision making
"""

    @staticmethod
    def hypothesis_generation_prompt(market_context: str, interaction_history: List) -> str:
        """Prompt for generating opponent hypotheses"""
        return f"""
Based on the market context and interaction history, generate 3 hypotheses about opponent strategies.

{market_context}

Recent interactions:
{interaction_history}

For each hypothesis, provide:
1. A description of the opponent's likely strategy
2. Predicted behavior patterns
3. How you would exploit this strategy

Respond with ONLY a JSON object in this exact format:
{{
  "hypotheses": [
    {{
      "description": "Detailed description of opponent strategy",
      "behavior_patterns": "Expected trading behavior",
      "exploitation": "How to exploit this strategy"
    }},
    {{
      "description": "Second hypothesis...",
      "behavior_patterns": "...",
      "exploitation": "..."
    }},
    {{
      "description": "Third hypothesis...",
      "behavior_patterns": "...",
      "exploitation": "..."
    }}
  ]
}}
"""


class AdaptiveAttributePrompts:
    """Prompts for LLM-designed attributes (Variance 3)"""

    @staticmethod
    def design_attributes_prompt(market_context: Dict[str, Any]) -> str:
        """Prompt for agent to design its own attributes - EMERGENT TRAITS VERSION"""
        market_conditions = ""
        if market_context:
            market_conditions = f"""
CURRENT MARKET CONDITIONS:
- Market volatility: {market_context['volatility']}
- Recent price trend: {market_context['trend']}
- Competition level: {market_context['competition']}
- Available liquidity: {market_context['liquidity']}
"""

        return f"""
You are a trading agent designing your own trading strategy through emergent parameter discovery.
{market_conditions}
TASK:
Create 4-8 strategy parameters that define YOUR OWN trading approach based on the market conditions above.

These are YOUR internal decision-making parameters, NOT beliefs about others.

PARAMETER DESIGN RULES:
1. Create parameter names that describe dimensions of YOUR decision-making
2. Values are not limited to 0-1, use whatever numeric range makes sense
3. Consider what decision rules would help you succeed in current market conditions
4. Think about how you want to respond to different market situations
5. Design parameters that capture YOUR strategic preferences
6. Focus on internal decision criteria, not observations about others

Respond with ONLY a JSON object with YOUR STRATEGY PARAMETERS (4-8 parameters):
{{
  "parameter_name_1": value,
  "parameter_name_2": value,
  "parameter_name_3": value,
  "parameter_name_4": value,
  "reasoning": "Brief explanation of your strategy and why these parameters matter"
}}
"""

    @staticmethod
    def adapt_attributes_prompt(
        current_attributes: Dict[str, float],
        performance_metrics: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> str:
        """Prompt for adapting attributes based on performance - EMERGENT TRAITS VERSION"""
        attr_display = "\n".join([
            f"- {attr.replace('_', ' ').title()}: {value:.2f}"
            for attr, value in current_attributes.items() if attr not in ['reasoning', 'confidence']
        ])

        market_conditions = ""
        if market_context:
            market_conditions = f"""
MARKET CONDITIONS:
- Volatility: {market_context['volatility']}
- Trend: {market_context['trend']}
- Competition: {market_context['competition']}
"""

        return f"""
Your current strategy parameters need adaptation based on performance feedback.

CURRENT PARAMETERS:
{attr_display}

PERFORMANCE METRICS:
- Current profit: ${performance_metrics['profit']}
- Win rate: {performance_metrics['win_rate']:.1%}
- Number of trades: {performance_metrics['trade_count']}
{market_conditions}
TASK:
Analyze your performance and adapt YOUR OWN strategy parameters.

These are YOUR internal decision rules, NOT beliefs about others.

ADAPTATION RULES:
1. You can modify existing parameter values (any numeric range)
2. You can create NEW parameters if you identify missing decision dimensions
3. You can remove parameters that aren't helping (by not including them in response)
4. Consider what strategic changes would improve YOUR performance
5. Base changes on concrete performance feedback

ANALYZE:
- What decision rules led to losses or missed opportunities?
- What market conditions did you misread?
- What parameters need adjustment?
- Are there new decision dimensions you need to add?

Respond with ONLY a JSON object with your ADAPTED PARAMETERS (include only the parameters you want to keep/modify/add):
{{
  "parameter_name_1": value,
  "parameter_name_2": value,
  "parameter_name_3": value,
  "reasoning": "Brief explanation of what changes you made and why they should improve performance"
}}
"""

    @staticmethod
    def infer_belief_traits_prompt(
        agent_id: str,
        event,
        agent_history: Dict[str, Any],
        market_state: Dict[str, Any],
        current_beliefs: Dict[str, float]
    ) -> str:
        """Prompt for inferring belief traits about another agent from observed behavior - EMERGENT TRAITS VERSION"""

        # Format event details
        event_type = event.event_type.value
        event_price = event.price if event.price else 'N/A'
        event_qty = event.quantity if event.quantity else 1

        # Format agent history
        total_trades = agent_history.get('total_trades', 0)
        last_bid = agent_history.get('last_bid_price') or 'None'
        last_ask = agent_history.get('last_ask_price') or 'None'
        recent_events = agent_history.get('recent_events', [])

        recent_events_str = "\n".join([
            f"  - {e.get('event_type', 'unknown')} at price {e.get('price', 'N/A')} (qty: {e.get('quantity', 1)})"
            for e in recent_events[-5:]
        ]) if recent_events else "  No recent events"

        # Format market state
        best_bid = market_state.get('current_best_bid') or 'N/A'
        best_ask = market_state.get('current_best_ask') or 'N/A'
        last_mkt_trade = market_state.get('last_trade_price') or 'N/A'
        spread = market_state.get('spread_width') or 'N/A'

        # Format current beliefs - show existing trait names
        if current_beliefs:
            belief_display = "\n".join([
                f"- {trait}: {value:.2f}"
                for trait, value in current_beliefs.items() if trait not in ['confidence', 'reasoning']
            ])
        else:
            belief_display = "No traits identified yet."

        return f"""
You are forming beliefs about what another trader THINKS and INTENDS through Theory of Mind reasoning.

CURRENT EVENT:
Agent {agent_id} just performed: {event_type} at price {event_price} (quantity: {event_qty})

AGENT'S TRADING HISTORY:
- Total trades completed: {total_trades}
- Last bid price: {last_bid}
- Last ask price: {last_ask}
- Recent activity:
{recent_events_str}

CURRENT MARKET STATE:
- Best bid: {best_bid}
- Best ask: {best_ask}
- Last market trade price: {last_mkt_trade}
- Bid-ask spread: {spread}

YOUR EXISTING BELIEFS ABOUT THIS AGENT:
{belief_display}

TASK:
Infer what this agent BELIEVES about the market and what they INTEND to do. These are YOUR BELIEFS about THEIR mental state.

BELIEF INFERENCE RULES:
1. Create belief dimensions that capture what THEY think, not what you think
2. CRITICAL: All dimension values MUST be NUMBERS (int or float) - NO strings
3. You can update existing belief dimensions or create new ones
4. Only infer beliefs you have evidence for from their actions
5. Focus on beliefs that would explain their trading decisions
6. Name dimensions based on what you're inferring about THEIR mind, not their behavior

Update your beliefs about what this agent thinks/intends.

Respond with ONLY a JSON object where ALL dimension values are NUMERIC:
{{
  "dimension_name": numeric_value,
  "another_dimension": numeric_value,
  "confidence": 0.X,
  "reasoning": "Brief explanation"
}}
"""

    @staticmethod
    def infer_discrete_beliefs_prompt(
        agent_id: str,
        event,
        agent_history: Dict[str, Any],
        market_state: Dict[str, Any],
        current_beliefs: Dict[str, Any]
    ) -> str:
        """Prompt for inferring discrete belief sets (GraphVar1)"""

        event_type = event.event_type.value
        event_price = event.price if event.price else 'N/A'
        event_qty = event.quantity if event.quantity else 1

        total_trades = agent_history.get('total_trades', 0)
        last_bid = agent_history.get('last_bid_price') or 'None'
        last_ask = agent_history.get('last_ask_price') or 'None'
        last_trade = agent_history.get('last_trade_price') or 'None'
        recent_events = agent_history.get('recent_events', [])

        recent_events_str = "\n".join([
            f"  - {e.get('event_type', 'unknown')} at price {e.get('price', 'N/A')} (qty: {e.get('quantity', 1)})"
            for e in recent_events[-5:]
        ]) if recent_events else "  No recent events"

        best_bid = market_state.get('current_best_bid') or 'N/A'
        best_ask = market_state.get('current_best_ask') or 'N/A'
        last_mkt_trade = market_state.get('last_trade_price') or 'N/A'
        spread = market_state.get('spread_width') or 'N/A'

        if current_beliefs:
            belief_display = "\n".join([
                f"- {key}: {value}"
                for key, value in current_beliefs.items() if key not in ['confidence', 'reasoning']
            ])
        else:
            belief_display = "No beliefs identified yet."

        return f"""
You are observing another trader's behavior to infer their possible states using discrete belief sets.

CURRENT EVENT:
Agent {agent_id} just performed: {event_type} at price {event_price} (quantity: {event_qty})

AGENT'S TRADING HISTORY:
- Total trades completed: {total_trades}
- Last bid price: {last_bid}
- Last ask price: {last_ask}
- Recent activity:
{recent_events_str}

CURRENT MARKET STATE:
- Best bid: {best_bid}
- Best ask: {best_ask}
- Last market trade price: {last_mkt_trade}
- Bid-ask spread: {spread}

EXISTING BELIEFS (Discrete Sets):
{belief_display}

TASK:
Based on what you observe, identify the POSSIBLE values for this agent's beliefs.
Use discrete sets to represent uncertainty about their true state.

For each belief category, list the possible values that are STILL PLAUSIBLE given this observation.

Respond with ONLY a JSON object with discrete sets:
{{
  "valuation": {{
    "possible_valuations": [85, 90, 95]
  }},
  "market_direction": {{
    "possible_directions": ["up", "down", "sideways"]
  }},
  "desperation_level": {{
    "possible_desperation": ["calm", "moderate", "desperate"]
  }},
  "cash_availability": {{
    "possible_cash": ["low", "medium", "high"]
  }},
  "exit_strategy": {{
    "possible_exits": ["hold_till_end", "sell_early", "opportunistic"]
  }},
  "confidence": 0.X,
  "reasoning": "Brief explanation of which possibilities were eliminated and why"
}}
"""

    @staticmethod
    def infer_probabilistic_beliefs_prompt(
        agent_id: str,
        event,
        agent_history: Dict[str, Any],
        market_state: Dict[str, Any],
        current_beliefs: Dict[str, Any]
    ) -> str:
        """Prompt for inferring probabilistic belief distributions (GraphVar2)"""

        event_type = event.event_type.value
        event_price = event.price if event.price else 'N/A'
        event_qty = event.quantity if event.quantity else 1

        total_trades = agent_history.get('total_trades', 0)
        last_bid = agent_history.get('last_bid_price') or 'None'
        last_ask = agent_history.get('last_ask_price') or 'None'
        last_trade = agent_history.get('last_trade_price') or 'None'
        recent_events = agent_history.get('recent_events', [])

        recent_events_str = "\n".join([
            f"  - {e.get('event_type', 'unknown')} at price {e.get('price', 'N/A')} (qty: {e.get('quantity', 1)})"
            for e in recent_events[-5:]
        ]) if recent_events else "  No recent events"

        best_bid = market_state.get('current_best_bid') or 'N/A'
        best_ask = market_state.get('current_best_ask') or 'N/A'
        last_mkt_trade = market_state.get('last_trade_price') or 'N/A'
        spread = market_state.get('spread_width') or 'N/A'

        if current_beliefs:
            belief_display = "\n".join([
                f"- {key}: {value}"
                for key, value in current_beliefs.items() if key not in ['confidence', 'reasoning']
            ])
        else:
            belief_display = "No beliefs identified yet."

        return f"""
You are observing another trader's behavior to infer probability distributions over their possible states.

CURRENT EVENT:
Agent {agent_id} just performed: {event_type} at price {event_price} (quantity: {event_qty})

AGENT'S TRADING HISTORY:
- Total trades completed: {total_trades}
- Last bid price: {last_bid}
- Last ask price: {last_ask}
- Recent activity:
{recent_events_str}

CURRENT MARKET STATE:
- Best bid: {best_bid}
- Best ask: {best_ask}
- Last market trade price: {last_mkt_trade}
- Bid-ask spread: {spread}

EXISTING BELIEFS (Probability Distributions):
{belief_display}

TASK:
Based on what you observe, update the probability distributions for this agent's beliefs.
Use Bayesian reasoning to assign probabilities to different possible states.

For each belief category, provide a probability distribution (must sum to 1.0).

Respond with ONLY a JSON object with probability distributions:
{{
  "valuation": {{
    "valuation_distribution": {{"85": 0.4, "90": 0.3, "95": 0.3}}
  }},
  "market_direction": {{
    "direction_distribution": {{"up": 0.3, "down": 0.4, "sideways": 0.3}}
  }},
  "desperation_level": {{
    "desperation_distribution": {{"calm": 0.6, "moderate": 0.3, "desperate": 0.1}}
  }},
  "cash_availability": {{
    "cash_distribution": {{"low": 0.5, "medium": 0.3, "high": 0.2}}
  }},
  "exit_strategy": {{
    "exit_distribution": {{"hold_till_end": 0.4, "sell_early": 0.3, "opportunistic": 0.3}}
  }},
  "confidence": 0.X,
  "reasoning": "Brief explanation of how you updated the probabilities based on this observation"
}}
"""


class TradingActionPrompts:
    """Final action decision prompts"""

    @staticmethod
    def unified_trading_prompt(
        market_context: str,
        belief_insights: str,
        trader_state: Dict[str, Any],
        cot_enabled: bool = False
    ) -> str:
        """Unified prompt for free trading (agent chooses BUY/SELL/WAIT based on inventory and market)"""
        cot_prefix = ChainOfThoughtPrompts.cot_reasoning_prefix() if cot_enabled else ""
        cot_suffix = "" if cot_enabled else ChainOfThoughtPrompts.no_cot_suffix()

        inventory = trader_state.get('inventory', 0)
        balance = trader_state.get('balance', 0)
        avg_purchase = trader_state.get('avg_purchase_price')

        inventory_context = ""
        if inventory > 0 and avg_purchase:
            inventory_context = f"""
INVENTORY STATUS:
- You are holding {inventory} unit(s)
- Average purchase price: ${avg_purchase:.2f}
- You can BUY more units (if you have cash) or SELL units you're holding
- Selling above ${avg_purchase:.2f} = profit per unit
- Selling below ${avg_purchase:.2f} = loss per unit
"""
        elif inventory > 0:
            inventory_context = f"""
INVENTORY STATUS:
- You are holding {inventory} unit(s)
- You can BUY more units (if you have cash) or SELL units you're holding
"""
        else:
            inventory_context = f"""
INVENTORY STATUS:
- You have NO inventory
- You can BUY units to start trading
- Current cash balance: ${balance:.0f}
"""

        return f"""
You are a proprietary trader with FULL FREEDOM to choose your actions.

{market_context}

{BasePrompts.MARKET_FUNDAMENTALS}

{belief_insights}

{inventory_context}

STRATEGIC THINKING:
- Active trading generates more opportunities than passive waiting
- Consider market momentum and where prices are heading
- Don't wait for the absolute perfect price - good enough is often better than perfect
- Trading velocity matters: completing trades quickly lets you capture new opportunities
- Strategic losses: taking a small loss now can free capital for bigger gains later
- Opportunity cost: holding inventory waiting for perfect prices means missing other trades
- Risk management: avoid spending your entire balance on one trade

YOUR GOAL: Maximize long-term profit through active, smart participation in the market.

{cot_prefix}

Respond with ONLY ONE of these actions:
"BUY [exact_price]" - to place a bid at that price (1 unit only)
"SELL [exact_price]" - to place an ask at that price (1 unit only)
"WAIT" - to wait for better conditions

{cot_suffix}
"""


class PromptBuilder:
    """Build complete prompts by combining scaffolding components"""

    @staticmethod
    def build_trading_prompt(
        agent_config: Dict[str, Any],
        market_context: str,
        trader_state: Dict[str, Any],
        belief_graph_data: Optional[Any] = None
    ) -> str:
        """
        Build a complete trading prompt based on agent configuration

        Args:
            agent_config: {
                'use_belief_graph': bool,
                'belief_format': 'json' | 'natural_language',
                'use_cot': bool,
                'graph_quality': 'perfect' | 'basic' | None,
                'job': 'Buy' | 'Sell'
            }
            market_context: Formatted market data string
            trader_state: Current trader state
            belief_graph_data: Belief graph insights (JSON or narrative)
        """

        # Build belief insights section
        if agent_config.get('use_belief_graph', False):
            if agent_config.get('belief_format') == 'json':
                belief_insights = BeliefGraphScaffolding.json_format().format(
                    belief_graph_json=belief_graph_data or "{}"
                )
            else:
                belief_insights = BeliefGraphScaffolding.natural_language_format().format(
                    belief_graph_narrative=belief_graph_data or "No beliefs available."
                )

            # Add graph quality context
            if agent_config.get('graph_quality') == 'perfect':
                belief_insights += "\n" + GraphQualityVariants.perfect_graph_context()
            elif agent_config.get('graph_quality') == 'basic':
                belief_insights += "\n" + GraphQualityVariants.basic_graph_context()
        else:
            belief_insights = GraphQualityVariants.no_graph_context()

        # Use unified trading prompt (free choice)
        return TradingActionPrompts.unified_trading_prompt(
            market_context,
            belief_insights,
            trader_state,
            agent_config.get('use_cot', False)
        )


class PromptParser:
    """Parse LLM responses consistently across all agents"""

    @staticmethod
    def parse_trading_action(response: str) -> Dict[str, Any]:
        """
        Parse trading action from LLM response (free choice: BUY/SELL/WAIT)

        Args:
            response: LLM response text

        Returns:
            {'action': str, 'price': Optional[int], 'reasoning': str}
        """
        import re

        response_upper = response.upper().strip()

        # Look for BUY with price (1 unit only)
        buy_match = re.search(r'BUY\s+(\d+)', response_upper)
        if buy_match:
            price = int(buy_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'BUY',
                'price': price,
                'reasoning': response
            }

        # Look for SELL with price (1 unit only)
        sell_match = re.search(r'SELL\s+(\d+)', response_upper)
        if sell_match:
            price = int(sell_match.group(1))
            price = max(1, min(500, price))
            return {
                'action': 'SELL',
                'price': price,
                'reasoning': response
            }

        # Check for WAIT
        if 'WAIT' in response_upper:
            return {
                'action': 'WAIT',
                'price': None,
                'reasoning': response
            }

        # No valid action found
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': f"Could not parse response: {response}"
        }

    @staticmethod
    def _extract_json_from_response(response: str) -> str:
        """Extract JSON object from response, handling markdown code fences and trailing commas"""
        import re

        # Strip markdown code fences if present
        code_fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_fence_match:
            response = code_fence_match.group(1)

        first_brace = response.find('{')
        if first_brace == -1:
            raise ValueError("No JSON object found in response")

        brace_count = 0
        start = first_brace

        for i, char in enumerate(response[first_brace:], start=first_brace):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = response[start:i+1]
                    # Remove trailing commas before closing braces (common LLM formatting error)
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    # Fix invalid escape sequences
                    json_str = PromptParser._fix_escape_sequences(json_str)
                    return json_str

        raise ValueError("Unmatched braces in JSON object")

    @staticmethod
    def _fix_escape_sequences(json_str: str) -> str:
        """Fix invalid escape sequences in JSON string"""
        import re

        # Replace invalid escapes with properly escaped versions
        # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
        # Find all backslashes followed by a character
        def fix_escape(match):
            char_after_backslash = match.group(1)
            # Keep valid JSON escapes
            if char_after_backslash in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                return match.group(0)
            # Escape the backslash for invalid escapes
            return '\\\\' + char_after_backslash

        return re.sub(r'\\(.)', fix_escape, json_str)

    @staticmethod
    def parse_attribute_design(response: str) -> Dict[str, float]:
        """Parse LLM-designed attributes from response - SUPPORTS EMERGENT TRAITS"""
        import json

        json_str = PromptParser._extract_json_from_response(response)
        data = json.loads(json_str)

        traits = {}
        for key, value in data.items():
            if value is None:
                continue
            key_lower = key.lower()
            if key_lower in ['reasoning', 'confidence']:
                traits[key_lower] = value
            else:
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Trait '{key}' must have numeric value, got: {type(value).__name__} - '{str(value)[:100]}'")
                traits[key] = float(value)

        return traits

    @staticmethod
    def parse_discrete_beliefs(response: str) -> Dict[str, Any]:
        """Parse discrete belief sets from LLM response (GraphVar1)"""
        import json

        json_str = PromptParser._extract_json_from_response(response)
        data = json.loads(json_str)
        return data

    @staticmethod
    def parse_probabilistic_beliefs(response: str) -> Dict[str, Any]:
        """Parse probabilistic belief distributions from LLM response (GraphVar2)"""
        import json

        json_str = PromptParser._extract_json_from_response(response)
        data = json.loads(json_str)
        return data
