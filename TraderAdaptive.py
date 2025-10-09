#!/usr/bin/env python3
"""
Adaptive Trader with Self-Designed Attributes

This trader class allows LLM agents to design and adapt their own trading attributes
that influence how they interact with the belief graph and make trading decisions.
"""

import os
import sys
import google.generativeai as genai
from typing import Dict, Any, Optional, Tuple
from agent_attributes import AttributeManager, AttributeDesigner
from llm_attribute_prompts import AttributePromptTemplates, AttributePromptParser
from agents.belief_graph import BeliefGraph, MarketEvent, EventType


class TraderAdaptive:
    """
    LLM-based trader that can design and adapt its own trading attributes.
    
    This trader integrates with the belief graph and uses its attributes to influence
    trading decisions and belief formation.
    """
    
    def __init__(self, ttype: str, tid: str, balance: float, params: Dict[str, Any], time: float):
        """
        Initialize the adaptive trader
        
        Args:
            ttype: Trader type identifier
            tid: Trader ID
            balance: Starting balance
            params: Trader parameters including API key and attribute settings
            time: Current time
        """
        self.ttype = ttype
        self.tid = tid
        self.balance = balance
        self.params = params or {}
        self.birthtime = time
        
        # Initialize attribute system
        self.attribute_manager = AttributeManager(tid)
        self.attributes_initialized = False
        
        # LLM configuration
        self.api_key = self.params.get('api_key') or os.getenv('GOOGLE_API_KEY')
        self.model_name = self.params.get('model_name', 'gemini-2.0-flash-lite')
        self.temperature = self.params.get('temperature', 0.3)
        self.max_tokens = self.params.get('max_tokens', 500)
        
        # Initialize LLM if API key is available
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"Initialized Adaptive Trader {tid} with model {self.model_name}")
        else:
            print(f"Warning: No API key provided for Adaptive Trader {tid}")
            self.model = None
        
        # Belief graph integration
        try:
            self.belief_graph = BeliefGraph(asset_id="BSE_ASSET")
            self.belief_graph.add_agent(tid)
        except ImportError:
            print(f"Warning: belief_graph module not found for trader {tid}")
            self.belief_graph = None
        
        # Trading state
        self.job = 'Buy'  # Buy or Sell mode
        self.last_purchase_price = None
        self.inventory = 0
        self.n_trades = 0
        
        # Performance tracking
        self.starting_balance = balance
        self.total_profit = 0.0
        self.trading_history = []
        
        # Attribute adaptation settings
        self.adaptation_enabled = self.params.get('adaptation_enabled', True)
        self.adaptation_interval = self.params.get('adaptation_interval', 10)  # Every 10 trades
        self.last_adaptation_check = 0
        
        # Market context tracking
        self.market_context = {
            'volatility': 0.0,
            'trend': 'unknown',
            'competition': 'unknown',
            'liquidity': 'unknown'
        }
    
    def initialize_attributes(self, market_context: Dict[str, Any] = None) -> None:
        """Initialize the agent's attributes using LLM decision making"""
        if self.attributes_initialized:
            return
        
        if not self.model:
            # Fallback: use balanced strategy if no LLM available
            self.attribute_manager.initialize_attributes("balanced")
            self.attributes_initialized = True
            return
        
        # Update market context
        if market_context:
            self.market_context.update(market_context)
        
        # Create initial design prompt
        available_strategies = AttributeDesigner.get_design_strategies()
        prompt = AttributePromptTemplates.create_initial_design_prompt(
            self.market_context, available_strategies
        )
        
        try:
            # Get LLM response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            # Parse response and initialize attributes
            strategy = AttributePromptParser.parse_design_response(response.text)
            self.attribute_manager.initialize_attributes(strategy)
            self.attributes_initialized = True
            
            print(f"Adaptive Trader {self.tid} designed attributes using strategy: {strategy}")
            
        except Exception as e:
            print(f"Error in attribute design for trader {self.tid}: {e}")
            # Fallback to balanced strategy
            self.attribute_manager.initialize_attributes("balanced")
            self.attributes_initialized = True
    
    def should_check_adaptation(self) -> bool:
        """Check if it's time to consider attribute adaptation"""
        if not self.adaptation_enabled:
            return False
        
        return self.n_trades >= self.last_adaptation_check + self.adaptation_interval
    
    def check_and_adapt_attributes(self, market_context: Dict[str, Any] = None) -> None:
        """Check if attributes should be adapted and perform adaptation if needed"""
        if not self.should_check_adaptation():
            return
        
        if not self.model or not self.attributes_initialized:
            return
        
        # Update market context
        if market_context:
            self.market_context.update(market_context)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Check if adaptation is needed
        if self.attribute_manager.adapt_attributes(performance_metrics):
            # Adaptation was performed automatically
            print(f"Trader {self.tid} automatically adapted attributes")
            return
        
        # If no automatic adaptation, ask LLM if manual adaptation is needed
        current_attributes = self.attribute_manager.get_attributes().to_dict()
        available_strategies = AttributeDesigner.get_design_strategies()
        
        prompt = AttributePromptTemplates.create_adaptation_prompt(
            current_attributes, performance_metrics, self.market_context, available_strategies
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            # Parse adaptation response
            strategy, reasoning = AttributePromptParser.parse_adaptation_response(response.text)
            
            # Apply adaptation
            self.attribute_manager.adapt_attributes(performance_metrics)
            print(f"Trader {self.tid} adapted attributes to {strategy}: {reasoning}")
            
        except Exception as e:
            print(f"Error in attribute adaptation for trader {self.tid}: {e}")
        
        # Update adaptation check timestamp
        self.last_adaptation_check = self.n_trades
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics for adaptation decisions"""
        current_profit = self.balance - self.starting_balance
        
        # Calculate market volatility (simplified)
        if len(self.trading_history) > 1:
            prices = [trade['price'] for trade in self.trading_history[-10:]]
            if len(prices) > 1:
                volatility = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices))) / len(prices)
                volatility = min(volatility / 100.0, 1.0)  # Normalize to 0-1
            else:
                volatility = 0.0
        else:
            volatility = 0.0
        
        # Calculate relative performance (simplified - could be enhanced with belief graph data)
        relative_performance = 0.0  # Placeholder - would compare to other agents
        
        return {
            'profit': current_profit,
            'market_volatility': volatility,
            'relative_performance': relative_performance,
            'trade_count': self.n_trades,
            'timestamp': self.birthtime
        }
    
    def update_belief_graph(self, market_event: MarketEvent) -> None:
        """Update the belief graph with market events"""
        if not self.belief_graph:
            return
        
        # Update belief graph
        self.belief_graph.update_beliefs(market_event)
        
        # Update market context based on belief graph insights
        self._update_market_context_from_beliefs()
    
    def _update_market_context_from_beliefs(self) -> None:
        """Update market context using belief graph insights"""
        if not self.belief_graph:
            return
        
        try:
            # Get current market state
            asset_node = self.belief_graph.asset_node
            if asset_node.current_best_bid and asset_node.current_best_ask:
                spread = asset_node.current_best_ask - asset_node.current_best_bid
                # High spread indicates low liquidity
                if spread > 10:
                    self.market_context['liquidity'] = 'low'
                elif spread < 5:
                    self.market_context['liquidity'] = 'high'
                else:
                    self.market_context['liquidity'] = 'moderate'
            
            # Update volatility from belief graph
            if asset_node.price_volatility > 0:
                self.market_context['volatility'] = min(asset_node.price_volatility / 100.0, 1.0)
            
        except Exception as e:
            print(f"Error updating market context from beliefs: {e}")
    
    def make_trading_decision(self, market_state: Dict[str, Any]) -> Tuple[str, Optional[float]]:
        """Make a trading decision influenced by the agent's attributes"""
        if not self.attributes_initialized:
            self.initialize_attributes(self.market_context)
        
        if not self.model:
            return "WAIT", None
        
        # Get current attributes
        attributes = self.attribute_manager.get_attributes().to_dict()
        
        # Get belief graph insights
        belief_graph_insights = self._get_belief_graph_insights()
        
        # Create attribute-influenced trading prompt
        prompt = AttributePromptTemplates.create_attribute_influenced_trading_prompt(
            attributes, market_state, belief_graph_insights
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            # Parse trading decision
            decision, price = AttributePromptParser.parse_trading_response(response.text)
            return decision, price
            
        except Exception as e:
            print(f"Error in trading decision for trader {self.tid}: {e}")
            return "WAIT", None
    
    def _get_belief_graph_insights(self) -> Dict[str, Any]:
        """Get insights from the belief graph for trading decisions"""
        if not self.belief_graph:
            return {}
        
        try:
            # Get current market state
            asset_node = self.belief_graph.asset_node
            
            # Analyze other agents' strategies
            agent_strategies = []
            for agent_id, agent_node in self.belief_graph.nodes.items():
                if agent_id != self.tid and hasattr(agent_node, 'strategy_type'):
                    if agent_node.strategy_type:
                        agent_strategies.append(f"Agent {agent_id}: {agent_node.strategy_type}")
            
            # Assess market sentiment
            sentiment = "neutral"
            if asset_node.price_volatility > 50:
                sentiment = "volatile"
            elif asset_node.price_volatility < 10:
                sentiment = "stable"
            
            # Risk assessment
            risk_level = "medium"
            if asset_node.spread_width and asset_node.spread_width > 15:
                risk_level = "high"
            elif asset_node.spread_width and asset_node.spread_width < 5:
                risk_level = "low"
            
            return {
                'agent_strategies': agent_strategies[:3],  # Top 3
                'market_sentiment': sentiment,
                'risk_assessment': risk_level
            }
            
        except Exception as e:
            print(f"Error getting belief graph insights: {e}")
            return {}
    
    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """Record a completed trade"""
        self.trading_history.append(trade_data)
        
        # Update belief graph with trade event
        if self.belief_graph:
            market_event = MarketEvent(
                event_id=f"trade_{len(self.trading_history)}",
                event_type=EventType.TRADE,
                timestamp=trade_data.get('time', 0),
                agent_id=self.tid,
                price=trade_data.get('price'),
                quantity=trade_data.get('quantity', 1),
                counterparty_id=trade_data.get('counterparty_id')
            )
            self.update_belief_graph(market_event)
        
        # Check if adaptation is needed
        self.check_and_adapt_attributes(self.market_context)
    
    def get_attributes_summary(self) -> Dict[str, Any]:
        """Get a summary of current attributes and adaptation history"""
        if not self.attributes_initialized:
            return {"error": "Attributes not initialized"}
        
        attributes = self.attribute_manager.get_attributes()
        return {
            'current_attributes': attributes.to_dict(),
            'design_strategy': self.attribute_manager.design_strategy,
            'adaptation_enabled': self.adaptation_enabled,
            'adaptation_history': self.attribute_manager.adapter.get_adaptation_history() if self.attribute_manager.adapter else [],
            'performance_metrics': self._calculate_performance_metrics()
        }
    
    def to_json(self) -> str:
        """Convert trader state to JSON"""
        return self.attribute_manager.to_json()
    
    @classmethod
    def from_json(cls, json_str: str, **kwargs) -> 'TraderAdaptive':
        """Create trader from JSON"""
        trader = cls(**kwargs)
        trader.attribute_manager = AttributeManager.from_json(json_str)
        trader.attributes_initialized = True
        return trader
