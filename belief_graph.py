"""
Explicit Belief Graph State Management for LLM Market Agents

This module implements the belief graph data structure for managing agent beliefs,
market state, and decision-making in multi-agent market simulations.

The belief graph tracks:
- Agent nodes with their inferred valuations and strategies
- Asset nodes with current market state
- Edges representing beliefs about other agents' valuations and intentions
- Probabilistic updates based on market events
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import math
import random
import logging

# Set up loggers for GraphVar1 and GraphVar2
gv1_logger = logging.getLogger('graphvar1_traders')
gv2_logger = logging.getLogger('graphvar2_traders')
bg_logger = logging.getLogger('belief_graph_traders')


class EventType(Enum):
    """Types of market events that can update the belief graph"""
    BID = "bid"
    ASK = "ask"
    TRADE = "trade"
    CANCEL = "cancel"
    AGENT_JOIN = "agent_join"
    AGENT_LEAVE = "agent_leave"


class NodeType(Enum):
    """Types of nodes in the belief graph"""
    AGENT = "agent"
    ASSET = "asset"


@dataclass
class MarketEvent:
    """Represents a market event that can update beliefs"""
    event_id: str
    event_type: EventType
    timestamp: float
    agent_id: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    counterparty_id: Optional[str] = None
    order_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'agent_id': self.agent_id,
            'price': self.price,
            'quantity': self.quantity,
            'counterparty_id': self.counterparty_id,
            'order_id': self.order_id
        }


@dataclass
class BeliefEdge:
    """Represents a belief relationship between nodes"""
    edge_id: str
    source_node: str
    target_node: str
    belief_type: str  # e.g., "valuation", "strategy", "intention"
    confidence: float  # 0.0 to 1.0
    value: Any  # The actual belief value (price, strategy type, etc.)
    timestamp: float
    evidence_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'edge_id': self.edge_id,
            'source_node': self.source_node,
            'target_node': self.target_node,
            'belief_type': self.belief_type,
            'confidence': self.confidence,
            'value': self.value,
            'timestamp': self.timestamp,
            'evidence_count': self.evidence_count
        }


@dataclass
class AgentNode:
    """Represents an agent in the belief graph"""
    agent_id: str
    node_type: NodeType = NodeType.AGENT
    last_bid_price: Optional[float] = None
    last_ask_price: Optional[float] = None
    last_trade_price: Optional[float] = None
    total_trades: int = 0
    total_volume: int = 0
    strategy_type: Optional[str] = None
    inferred_valuation: Optional[float] = None
    valuation_confidence: float = 0.0
    aggressiveness_score: float = 0.0  # -1.0 (passive) to 1.0 (aggressive)
    last_activity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'agent_id': self.agent_id,
            'node_type': self.node_type.value,
            'last_bid_price': self.last_bid_price,
            'last_ask_price': self.last_ask_price,
            'last_trade_price': self.last_trade_price,
            'total_trades': self.total_trades,
            'total_volume': self.total_volume,
            'strategy_type': self.strategy_type,
            'inferred_valuation': self.inferred_valuation,
            'valuation_confidence': self.valuation_confidence,
            'aggressiveness_score': self.aggressiveness_score,
            'last_activity': self.last_activity
        }


@dataclass
class AssetNode:
    """Represents the traded asset in the belief graph"""
    asset_id: str
    node_type: NodeType = NodeType.ASSET
    current_best_bid: Optional[float] = None
    current_best_ask: Optional[float] = None
    last_trade_price: Optional[float] = None
    volume_traded: int = 0
    price_volatility: float = 0.0
    spread_width: Optional[float] = None
    market_depth_bid: int = 0
    market_depth_ask: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'asset_id': self.asset_id,
            'node_type': self.node_type.value,
            'current_best_bid': self.current_best_bid,
            'current_best_ask': self.current_best_ask,
            'last_trade_price': self.last_trade_price,
            'volume_traded': self.volume_traded,
            'price_volatility': self.price_volatility,
            'spread_width': self.spread_width,
            'market_depth_bid': self.market_depth_bid,
            'market_depth_ask': self.market_depth_ask
        }


class BeliefGraph:
    """
    Main belief graph class for managing agent beliefs and market state.
    
    The belief graph maintains:
    - Nodes for each agent and the traded asset
    - Edges representing beliefs about other agents' valuations and strategies
    - Probabilistic updates based on market events
    - Query interface for decision-making
    """
    
    def __init__(self, asset_id: str = "DEFAULT_ASSET"):
        self.graph_id = str(uuid.uuid4())
        self.asset_id = asset_id
        self.nodes: Dict[str, AgentNode | AssetNode] = {}
        self.edges: Dict[str, BeliefEdge] = {}
        self.event_history: List[MarketEvent] = []
        self.current_time = 0.0
        
        # Initialize the asset node
        self.asset_node = AssetNode(asset_id=asset_id)
        self.nodes[asset_id] = self.asset_node
        
        # Belief update parameters
        self.valuation_decay_rate = 0.95  # How quickly old valuation beliefs decay
        self.confidence_boost = 0.1  # How much confidence increases with new evidence
        self.max_confidence = 0.95  # Maximum confidence level
        
    def add_agent(self, agent_id: str) -> None:
        """Add a new agent to the belief graph"""
        if agent_id not in self.nodes:
            agent_node = AgentNode(agent_id=agent_id)
            self.nodes[agent_id] = agent_node
            
            # Add initial beliefs about this agent
            self._add_initial_beliefs(agent_id)
    
    def _add_initial_beliefs(self, agent_id: str) -> None:
        """Add initial beliefs about a new agent"""
        # Add belief about agent's strategy (initially unknown)
        strategy_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="strategy",
            confidence=0.1,
            value="unknown",
            timestamp=self.current_time
        )
        self.edges[strategy_edge.edge_id] = strategy_edge
        
        # Add belief about agent's valuation (initially unknown)
        valuation_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="valuation",
            confidence=0.1,
            value=None,
            timestamp=self.current_time
        )
        self.edges[valuation_edge.edge_id] = valuation_edge
    
    def update_beliefs(self, event: MarketEvent) -> None:
        """
        Update the belief graph based on a market event.
        
        This is the core function that ingests market events and revises
        the belief graph accordingly.
        """
        self.current_time = event.timestamp
        self.event_history.append(event)
        
        # Ensure the agent exists in the graph
        if event.agent_id and event.agent_id not in self.nodes:
            self.add_agent(event.agent_id)
        
        # Update based on event type
        if event.event_type == EventType.BID:
            self._update_beliefs_from_bid(event)
        elif event.event_type == EventType.ASK:
            self._update_beliefs_from_ask(event)
        elif event.event_type == EventType.TRADE:
            self._update_beliefs_from_trade(event)
        elif event.event_type == EventType.CANCEL:
            self._update_beliefs_from_cancel(event)
        
        # Update asset state
        self._update_asset_state()
        
        # Decay old beliefs
        self._decay_old_beliefs()
    
    def _update_beliefs_from_bid(self, event: MarketEvent) -> None:
        """Update beliefs based on a bid event"""
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_bid_price = event.price
        agent_node.last_activity = event.timestamp
        
        # Update valuation belief
        self._update_valuation_belief(event.agent_id, event.price, "bid")
    
    def _update_beliefs_from_ask(self, event: MarketEvent) -> None:
        """Update beliefs based on an ask event"""
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_ask_price = event.price
        agent_node.last_activity = event.timestamp
        
        # Update valuation belief
        self._update_valuation_belief(event.agent_id, event.price, "ask")
    
    def _update_beliefs_from_trade(self, event: MarketEvent) -> None:
        """Update beliefs based on a trade event"""
        bg_logger.debug(f"[BG-DEBUG] _update_beliefs_from_trade called for agent {event.agent_id} at price {event.price}")
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        old_aggr = agent_node.aggressiveness_score
        agent_node.last_trade_price = event.price
        agent_node.total_trades += 1
        agent_node.total_volume += event.quantity or 1
        agent_node.last_activity = event.timestamp
        
        # Update valuation belief with high confidence (actual trade)
        self._update_valuation_belief(event.agent_id, event.price, "trade", high_confidence=True)
        # Update aggressiveness based on trade price compared to previous trades
        if self.asset_node.last_trade_price and agent_node.total_trades > 1:
            price_ratio = event.price / self.asset_node.last_trade_price
            bg_logger.debug(f"[BG-TRADE-AGGR] Agent {event.agent_id}: price={event.price}, last_price={self.asset_node.last_trade_price}, ratio={price_ratio:.3f}")
            
            # Buyer perspective (assuming agent_id is buyer in BSE)
            if price_ratio > 1.01:  # Paid >1% more than last trade
                agent_node.aggressiveness_score = min(1.0, agent_node.aggressiveness_score + 0.3)
                bg_logger.debug(f"[BG-TRADE-AGGR] AGGRESSIVE buyer: paid {price_ratio-1:.1%} more")
            elif price_ratio < 0.99:  # Paid <1% less than last trade  
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.2)
                bg_logger.debug(f"[BG-TRADE-AGGR] PASSIVE buyer: paid {1-price_ratio:.1%} less")
            else:
                # Near market price - slight shift toward passive
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.05)
                bg_logger.debug(f"[BG-TRADE-AGGR] NEUTRAL trade")
        else:
            # First trade - random initial aggressiveness
            import random
            agent_node.aggressiveness_score = random.uniform(-0.3, 0.3)
            bg_logger.debug(f"[BG-TRADE-AGGR] First trade, random initial aggr={agent_node.aggressiveness_score:.3f}")
        
        bg_logger.debug(f"[BG-TRADE-AGGR] Agent {event.agent_id} aggr: {old_aggr:.3f} -> {agent_node.aggressiveness_score:.3f}")
    
        # Update strategy belief based on trade
        self._update_strategy_belief(event.agent_id, "trade", event.price)
        
        # Also update counterparty if available
        if event.counterparty_id and event.counterparty_id in self.nodes:
            seller_node = self.nodes[event.counterparty_id]
            old_seller_aggr = seller_node.aggressiveness_score
            seller_node.last_trade_price = event.price
            seller_node.total_trades += 1
            seller_node.total_volume += event.quantity or 1
            
            # Seller perspective - opposite of buyer
            if self.asset_node.last_trade_price and seller_node.total_trades > 1:
                price_ratio = event.price / self.asset_node.last_trade_price
                
                if price_ratio < 0.99:  # Sold <1% below last trade
                    seller_node.aggressiveness_score = min(1.0, seller_node.aggressiveness_score + 0.3)
                    bg_logger.debug(f"[BG-TRADE-AGGR] AGGRESSIVE seller {event.counterparty_id}: sold {1-price_ratio:.1%} below")
                elif price_ratio > 1.01:  # Sold >1% above last trade
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.2)
                    bg_logger.debug(f"[BG-TRADE-AGGR] PASSIVE seller {event.counterparty_id}: sold {price_ratio-1:.1%} above")
                else:
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.05)
            else:
                # First trade - random initial
                import random
                seller_node.aggressiveness_score = random.uniform(-0.3, 0.3)
                
            bg_logger.debug(f"[BG-TRADE-AGGR] Seller {event.counterparty_id} aggr: {old_seller_aggr:.3f} -> {seller_node.aggressiveness_score:.3f}")
            
            # Update seller strategy
            self._update_valuation_belief(event.counterparty_id, event.price, "trade", high_confidence=True)
            self._update_strategy_belief(event.counterparty_id, "trade", event.price)
        
        # Update asset state
        self.asset_node.last_trade_price = event.price
        self.asset_node.volume_traded += event.quantity or 1
    
    def _update_beliefs_from_cancel(self, event: MarketEvent) -> None:
        """Update beliefs based on a cancel event"""
        if not event.agent_id:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_activity = event.timestamp
        
        # Cancellation might indicate uncertainty or strategy change
        agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.05)
    
    def _update_valuation_belief(self, agent_id: str, price: float, action_type: str, high_confidence: bool = False) -> None:
        """Update the belief about an agent's valuation"""
        # Find existing valuation edge
        valuation_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "valuation"):
                valuation_edge = edge
                break
        
        if valuation_edge is None:
            # Create new valuation edge
            valuation_edge = BeliefEdge(
                edge_id=str(uuid.uuid4()),
                source_node=self.asset_id,
                target_node=agent_id,
                belief_type="valuation",
                confidence=0.3 if high_confidence else 0.1,
                value=price,
                timestamp=self.current_time
            )
            self.edges[valuation_edge.edge_id] = valuation_edge
        else:
            # Update existing valuation belief
            old_value = valuation_edge.value
            old_confidence = valuation_edge.confidence
            
            # Weighted average of old and new values
            if old_value is not None:
                if high_confidence:
                    # Trade events get higher weight
                    new_value = 0.7 * price + 0.3 * old_value
                    new_confidence = min(self.max_confidence, old_confidence + 0.3)
                else:
                    # Bid/ask events get lower weight
                    new_value = 0.3 * price + 0.7 * old_value
                    new_confidence = min(self.max_confidence, old_confidence + 0.1)
            else:
                new_value = price
                new_confidence = 0.3 if high_confidence else 0.1
            
            valuation_edge.value = new_value
            valuation_edge.confidence = new_confidence
            valuation_edge.timestamp = self.current_time
            valuation_edge.evidence_count += 1
            
            # Sync the agent node's valuation fields
            agent_node = self.nodes[agent_id]
            agent_node.inferred_valuation = new_value
            agent_node.valuation_confidence = new_confidence
    
    def _update_strategy_belief(self, agent_id: str, action_type: str, price: float) -> None:
        """Update the belief about an agent's strategy"""
        bg_logger.debug(f"[BG-STRAT-START] Updating strategy for {agent_id}, action={action_type}, price={price}")
        
        # Find existing strategy edge
        strategy_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "strategy"):
                strategy_edge = edge
                break
        
        if strategy_edge is None:
            bg_logger.debug(f"[BG-STRAT-ERROR] No strategy edge found for {agent_id}")
            return
        
        # Simple strategy classification based on behavior patterns
        agent_node = self.nodes[agent_id]
        
        bg_logger.debug(f"[BG-STRAT-INFO] Agent {agent_id}: trades={agent_node.total_trades}, aggr_score={agent_node.aggressiveness_score:.3f}, current_strategy={agent_node.strategy_type}")
        
        # Always classify based on aggressiveness, even if no trades yet
        old_strategy = agent_node.strategy_type
        
        # Use more sensitive thresholds
        if agent_node.aggressiveness_score > 0.15:
            strategy = "aggressive"
        elif agent_node.aggressiveness_score < -0.15:
            strategy = "passive"
        else:
            strategy = "neutral"
        
        bg_logger.debug(f"[BG-STRAT-CLASSIFY] Agent {agent_id}: aggr={agent_node.aggressiveness_score:.3f} -> strategy={strategy} (was {old_strategy})")
        
        strategy_edge.value = strategy
        strategy_edge.confidence = min(self.max_confidence, strategy_edge.confidence + 0.1)
        strategy_edge.timestamp = self.current_time
        strategy_edge.evidence_count += 1
        
        # Sync the agent node's strategy_type field
        agent_node.strategy_type = strategy
        
        bg_logger.debug(f"[BG-STRAT-END] Agent {agent_id} strategy updated to {strategy} (confidence={strategy_edge.confidence:.2f})")
    
    def _update_asset_state(self) -> None:
        """Update the asset node state based on current market conditions"""
        # This would typically be called with actual market data
        # For now, we'll update based on the belief graph state
        
        # Calculate spread if we have both bid and ask
        if (self.asset_node.current_best_bid is not None and 
            self.asset_node.current_best_ask is not None):
            self.asset_node.spread_width = self.asset_node.current_best_ask - self.asset_node.current_best_bid
    
    def _decay_old_beliefs(self) -> None:
        """Decay confidence in old beliefs"""
        current_time = self.current_time
        for edge in self.edges.values():
            time_diff = current_time - edge.timestamp
            if time_diff > 100:  # Decay beliefs older than 100 time units
                decay_factor = self.valuation_decay_rate ** (time_diff / 100)
                edge.confidence *= decay_factor
    
    def query_action(self, agent_id: str, current_market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the belief graph for decision-making.
        
        This function serializes the belief graph and returns it in a format
        suitable for LLM processing.
        """
        # Update asset state with current market data
        if 'best_bid' in current_market_state:
            self.asset_node.current_best_bid = current_market_state['best_bid']
        if 'best_ask' in current_market_state:
            self.asset_node.current_best_ask = current_market_state['best_ask']
        if 'last_trade' in current_market_state:
            self.asset_node.last_trade_price = current_market_state['last_trade']
        
        # Prepare the belief graph for LLM consumption
        belief_graph_data = {
            'graph_id': self.graph_id,
            'current_time': self.current_time,
            'asset_state': self.asset_node.to_dict(),
            'agents': {},
            'beliefs': [],
            'recent_events': []
        }
        
        # Add agent information
        for node_id, node in self.nodes.items():
            if isinstance(node, AgentNode):
                belief_graph_data['agents'][node_id] = node.to_dict()
        
        # Add belief edges
        for edge in self.edges.values():
            belief_graph_data['beliefs'].append(edge.to_dict())
        
        # Add recent events (last 10)
        recent_events = self.event_history[-10:] if len(self.event_history) > 10 else self.event_history
        belief_graph_data['recent_events'] = [event.to_dict() for event in recent_events]
        
        # Add strategic insights
        belief_graph_data['strategic_insights'] = self._generate_strategic_insights(agent_id)
        
        return belief_graph_data
    
    def _generate_strategic_insights(self, agent_id: str) -> Dict[str, Any]:
        """Generate strategic insights for the querying agent"""
        insights = {
            'competitors': [],
            'market_opportunities': [],
            'risk_factors': []
        }
        
        # Analyze competitors
        for node_id, node in self.nodes.items():
            if isinstance(node, AgentNode) and node_id != agent_id:
                competitor_info = {
                    'agent_id': node_id,
                    'strategy': node.strategy_type or "unknown",
                    'aggressiveness': node.aggressiveness_score,
                    'valuation_estimate': node.inferred_valuation,
                    'confidence': node.valuation_confidence,
                    'recent_activity': node.last_activity
                }
                insights['competitors'].append(competitor_info)
        
        # Identify market opportunities
        if (self.asset_node.current_best_bid is not None and 
            self.asset_node.current_best_ask is not None):
            spread = self.asset_node.current_best_ask - self.asset_node.current_best_bid
            if spread > 5:  # Arbitrage opportunity
                insights['market_opportunities'].append({
                    'type': 'arbitrage',
                    'spread': spread,
                    'description': f"Large spread of {spread} points"
                })
        
        # Identify risk factors
        if self.asset_node.price_volatility > 0.1:
            insights['risk_factors'].append({
                'type': 'high_volatility',
                'value': self.asset_node.price_volatility,
                'description': "High price volatility detected"
            })
        
        return insights
    
    def to_json(self) -> str:
        """Serialize the belief graph to JSON"""
        graph_data = {
            'graph_id': self.graph_id,
            'asset_id': self.asset_id,
            'current_time': self.current_time,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            'event_history': [event.to_dict() for event in self.event_history[-50:]]  # Last 50 events
        }
        return json.dumps(graph_data, indent=2)
    
    def from_json(self, json_str: str) -> None:
        """Deserialize the belief graph from JSON"""
        data = json.loads(json_str)
        self.graph_id = data['graph_id']
        self.asset_id = data['asset_id']
        self.current_time = data['current_time']
        
        # Reconstruct nodes
        self.nodes.clear()
        for node_id, node_data in data['nodes'].items():
            if node_data['node_type'] == NodeType.AGENT.value:
                self.nodes[node_id] = AgentNode(**node_data)
            elif node_data['node_type'] == NodeType.ASSET.value:
                self.nodes[node_id] = AssetNode(**node_data)
        
        # Reconstruct edges
        self.edges.clear()
        for edge_id, edge_data in data['edges'].items():
            self.edges[edge_id] = BeliefEdge(**edge_data)
        
        # Reconstruct event history
        self.event_history = [MarketEvent(**event_data) for event_data in data['event_history']]
    
    def update_agent_attributes(self, agent_id: str, attributes: Dict[str, Any]) -> None:
        """
        Update agent attributes in the belief graph based on AI-designed attributes.
        
        This method allows the adaptive trader to update the belief graph
        with its designed trading personality attributes.
        """
        if agent_id not in self.nodes:
            self.add_agent(agent_id)
        
        agent_node = self.nodes[agent_id]
        
        # Update aggressiveness score based on AI-designed aggressiveness
        if 'aggressiveness' in attributes:
            # Convert from 0-1 scale to -1 to 1 scale for belief graph
            ai_aggressiveness = attributes['aggressiveness']
            belief_aggressiveness = (ai_aggressiveness * 2) - 1  # 0->-1, 0.5->0, 1->1
            agent_node.aggressiveness_score = belief_aggressiveness
        
        # Update strategy type based on AI's design choice
        if 'design_strategy' in attributes:
            agent_node.strategy_type = attributes['design_strategy']
        
        # Update last activity timestamp
        agent_node.last_activity = self.current_time
        
        # Add belief edge about the agent's strategy
        strategy_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="strategy",
            confidence=0.9,  # High confidence since AI designed it
            value=attributes.get('design_strategy', 'adaptive'),
            timestamp=self.current_time
        )
        self.edges[strategy_edge.edge_id] = strategy_edge
        
        # Add belief edge about the agent's aggressiveness
        aggressiveness_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="aggressiveness",
            confidence=0.9,  # High confidence since AI designed it
            value=attributes.get('aggressiveness', 0.5),
            timestamp=self.current_time
        )
        self.edges[aggressiveness_edge.edge_id] = aggressiveness_edge
    
    def get_agent_beliefs(self, agent_id: str) -> Dict[str, Any]:
        """Get all beliefs about a specific agent"""
        beliefs = {}
        
        # First get the raw belief edges
        for edge in self.edges.values():
            if edge.target_node == agent_id:
                if edge.belief_type not in beliefs:
                    beliefs[edge.belief_type] = []
                beliefs[edge.belief_type].append(edge.to_dict())
        
        # Extract valuation estimate if available
        valuation_estimate = None
        if agent_id in self.nodes:
            agent_node = self.nodes[agent_id]
            if hasattr(agent_node, 'inferred_valuation'):
                valuation_estimate = agent_node.inferred_valuation
            
            # Also try to get from valuation edge
            for edge in self.edges.values():
                if (edge.target_node == agent_id and 
                    edge.belief_type == "valuation" and 
                    edge.value and isinstance(edge.value, dict)):
                    possible_vals = edge.value.get("possible_valuations", [])
                    if possible_vals:
                        # Use median of possible valuations as estimate
                        valuation_estimate = sorted(possible_vals)[len(possible_vals)//2]
                        break
        
        # Extract strategy type if available
        strategy_type = None
        if agent_id in self.nodes:
            agent_node = self.nodes[agent_id]
            if hasattr(agent_node, 'strategy_type'):
                strategy_type = agent_node.strategy_type
        
        # Add simplified summary
        beliefs['valuation_estimate'] = valuation_estimate
        beliefs['strategy_type'] = strategy_type
        
        return beliefs
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get a summary of the current market state"""
        return {
            'asset_state': self.asset_node.to_dict(),
            'active_agents': len([n for n in self.nodes.values() if isinstance(n, AgentNode)]),
            'total_beliefs': len(self.edges),
            'recent_events': len(self.event_history),
            'current_time': self.current_time
        }



class GraphVar1:
    """
    Main belief graph class for managing agent beliefs and market state using discrete belief sets.
    
    The belief graph maintains:
    - Nodes for each agent and the traded asset
    - Edges representing discrete belief sets about other agents' valuations and strategies
    - Set elimination updates based on market events
    - Query interface for decision-making
    """
    
    def __init__(self, asset_id: str = "DEFAULT_ASSET"):
        self.graph_id = str(uuid.uuid4())
        self.asset_id = asset_id
        self.nodes: Dict[str, AgentNode | AssetNode] = {}
        self.edges: Dict[str, BeliefEdge] = {}
        self.event_history: List[MarketEvent] = []
        self.current_time = 0.0
        
        # Initialize the asset node
        self.asset_node = AssetNode(asset_id=asset_id)
        self.nodes[asset_id] = self.asset_node
        
        # Belief update parameters - CONFIGURABLE, NO HARDCODING!
        self.valuation_decay_rate = 0.95  # How quickly old valuation beliefs decay
        self.confidence_boost = 0.1  # How much confidence increases with new evidence
        self.max_confidence = 0.95  # Maximum confidence level
        
        # GraphVar1 specific thresholds - make configurable
        self.bid_valuation_buffer = 5  # Buffer below bid price for valuation elimination
        self.ask_valuation_buffer = 5  # Buffer above ask price for valuation elimination
        self.trade_valuation_margin = 10  # Wider margin for trades to avoid over-elimination
        self.aggressive_bid_threshold = 0.98  # Within 2% of ask = aggressive
        self.aggressive_ask_threshold = 1.02  # Within 2% of bid = aggressive
        self.large_volume_threshold = 5  # Volume threshold for cash inference
        self.strategy_aggr_threshold = 0.15  # Aggressiveness threshold for strategy classification
        
    def add_agent(self, agent_id: str) -> None:
        """Add a new agent to the belief graph"""
        if agent_id not in self.nodes:
            agent_node = AgentNode(agent_id=agent_id)
            self.nodes[agent_id] = agent_node
            
            # Add initial beliefs about this agent
            self._add_initial_beliefs(agent_id)
    
    def _add_initial_beliefs(self, agent_id: str) -> None:
        """Add initial discrete belief sets about a new agent"""
        # Add discrete belief about agent's valuation possibilities
        valuation_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="valuation",
            confidence=1.0,  # Full confidence in the discrete set
            value={"possible_valuations": [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]},
            timestamp=self.current_time
        )
        self.edges[valuation_edge.edge_id] = valuation_edge
        
        # Add discrete belief about agent's market direction
        direction_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="market_direction",
            confidence=1.0,
            value={"possible_directions": ["up", "down", "sideways"]},
            timestamp=self.current_time
        )
        self.edges[direction_edge.edge_id] = direction_edge
        
        # Add discrete belief about agent's desperation level
        desperation_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="desperation_level",
            confidence=1.0,
            value={"possible_desperation": ["calm", "moderate", "desperate"]},
            timestamp=self.current_time
        )
        self.edges[desperation_edge.edge_id] = desperation_edge
        
        # Add discrete belief about agent's available cash
        cash_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="available_cash",
            confidence=1.0,
            value={"possible_cash": ["low", "medium", "high"]},
            timestamp=self.current_time
        )
        self.edges[cash_edge.edge_id] = cash_edge
        
        # Add discrete belief about agent's exit strategy
        exit_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="exit_strategy",
            confidence=1.0,
            value={"possible_exits": ["hold_till_end", "sell_early", "opportunistic"]},
            timestamp=self.current_time
        )
        self.edges[exit_edge.edge_id] = exit_edge
        
        # Add self-beliefs for this agent (beliefs about own state)
        # These will be dynamically updated based on market observations
        
        # Self-belief about market direction assessment
        self_direction_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=agent_id,  # Agent has beliefs about themselves
            target_node=agent_id,
            belief_type="self_market_direction",
            confidence=1.0,
            value={"possible_directions": ["up", "down", "sideways"]},
            timestamp=self.current_time
        )
        self.edges[self_direction_edge.edge_id] = self_direction_edge
        
        # Self-belief about optimal entry prices
        self_price_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=agent_id,
            target_node=agent_id,
            belief_type="self_optimal_entry",
            confidence=1.0,
            value={"possible_prices": [75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]},
            timestamp=self.current_time
        )
        self.edges[self_price_edge.edge_id] = self_price_edge
        
        # Self-belief about time urgency
        self_urgency_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=agent_id,
            target_node=agent_id,
            belief_type="self_time_urgency",
            confidence=1.0,
            value={"possible_urgency": ["low", "medium", "high"]},
            timestamp=self.current_time
        )
        self.edges[self_urgency_edge.edge_id] = self_urgency_edge
        
    def _update_discrete_beliefs_from_market_event(self, agent_id: str, price: float, action_type: str, high_confidence: bool = False) -> None:
        """Update discrete belief sets using set elimination logic based on market events"""
        gv1_logger.debug(f"[BG-DISCRETE] Updating discrete beliefs for {agent_id}, price={price}, action={action_type}, high_conf={high_confidence}")
        
        # Update valuation beliefs using set elimination
        valuation_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "valuation"):
                valuation_edge = edge
                break
        
        if valuation_edge and valuation_edge.value and "possible_valuations" in valuation_edge.value:
            possible_vals = valuation_edge.value["possible_valuations"]
            
            # Set elimination based on observed price and action
            if action_type == "bid":
                # Agent bidding at price X suggests valuation >= X
                # Eliminate valuations significantly below bid price
                threshold = price - self.bid_valuation_buffer  # Configurable buffer
                old_vals = possible_vals.copy()
                possible_vals = [v for v in possible_vals if v >= threshold]
                if old_vals != possible_vals:  # Only log if there was a change
                    gv1_logger.info(f"[GraphVar1-BELIEF-UPDATE] {agent_id} BID at {price}:")
                    gv1_logger.info(f"  BEFORE: {old_vals} ({len(old_vals)} possibilities)")
                    gv1_logger.info(f"  AFTER:  {possible_vals} ({len(possible_vals)} possibilities)")
                    gv1_logger.info(f"  ELIMINATED: valuations < {threshold} (price - buffer={self.bid_valuation_buffer})")
            elif action_type == "ask":
                # Agent asking at price X suggests valuation <= X
                # Eliminate valuations significantly above ask price  
                threshold = price + self.ask_valuation_buffer  # Configurable buffer
                old_vals = possible_vals.copy()
                possible_vals = [v for v in possible_vals if v <= threshold]
                if old_vals != possible_vals:  # Only log if there was a change
                    gv1_logger.info(f"[GraphVar1-BELIEF-UPDATE] {agent_id} ASK at {price}:")
                    gv1_logger.info(f"  BEFORE: {old_vals} ({len(old_vals)} possibilities)")
                    gv1_logger.info(f"  AFTER:  {possible_vals} ({len(possible_vals)} possibilities)")
                    gv1_logger.info(f"  ELIMINATED: valuations > {threshold} (price + buffer={self.ask_valuation_buffer})")
            elif action_type == "trade":
                # Trade at price X suggests valuation very close to X
                if high_confidence:
                    # Keep valuations within tight range of trade price
                    margin = self.trade_valuation_margin  # Configurable margin
                    old_vals = possible_vals.copy()
                    possible_vals = [v for v in possible_vals if abs(v - price) <= margin]
                    if old_vals != possible_vals:  # Only log if there was a change
                        gv1_logger.info(f"[GraphVar1-BELIEF-UPDATE] {agent_id} TRADE at {price} (HIGH CONFIDENCE):")
                        gv1_logger.info(f"  BEFORE: {old_vals} ({len(old_vals)} possibilities)")
                        gv1_logger.info(f"  AFTER:  {possible_vals} ({len(possible_vals)} possibilities)")
                        gv1_logger.info(f"  ELIMINATED: outside range [{price-margin}, {price+margin}]")
            
            # Update the discrete set
            valuation_edge.value["possible_valuations"] = possible_vals
            valuation_edge.timestamp = self.current_time
        
        # Update desperation level based on price aggressiveness
        desperation_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "desperation_level"):
                desperation_edge = edge
                break
        
        if desperation_edge and "possible_desperation" in desperation_edge.value:
            possible_desp = desperation_edge.value["possible_desperation"]
            
            # Get market context for aggressiveness assessment
            current_bid = self.asset_node.current_best_bid
            current_ask = self.asset_node.current_best_ask
            
            if action_type == "bid" and current_ask:
                # Aggressive bidding near ask price suggests desperation
                if price >= current_ask * 0.98:  # Bidding within 2% of ask
                    possible_desp = [d for d in possible_desp if d != "calm"]
                    gv1_logger.debug(f"[BG-DISCRETE] Aggressive bid: eliminated 'calm' desperation, remaining: {possible_desp}")
            elif action_type == "ask" and current_bid:
                # Aggressive asking near bid price suggests desperation
                if price <= current_bid * 1.02:  # Asking within 2% of bid
                    possible_desp = [d for d in possible_desp if d != "calm"]
                    gv1_logger.debug(f"[BG-DISCRETE] Aggressive ask: eliminated 'calm' desperation, remaining: {possible_desp}")
            
            desperation_edge.value["possible_desperation"] = possible_desp
            desperation_edge.timestamp = self.current_time
        
        # Update available cash based on trade volume patterns
        agent_node = self.nodes[agent_id]
        if agent_node.total_volume > 0:  # Only update if we have volume data
            cash_edge = None
            for edge in self.edges.values():
                if (edge.source_node == self.asset_id and 
                    edge.target_node == agent_id and 
                    edge.belief_type == "available_cash"):
                    cash_edge = edge
                    break
            
            if cash_edge and "possible_cash" in cash_edge.value:
                possible_cash = cash_edge.value["possible_cash"]
                
                # Large trades suggest higher available cash
                if agent_node.total_volume >= 5:  # Arbitrary threshold for "large" volume
                    possible_cash = [c for c in possible_cash if c != "low"]
                    gv1_logger.debug(f"[BG-DISCRETE] High volume ({agent_node.total_volume}): eliminated 'low' cash, remaining: {possible_cash}")
                
                cash_edge.value["possible_cash"] = possible_cash
                cash_edge.timestamp = self.current_time
    
    def update_beliefs(self, event: MarketEvent) -> None:
        """
        Update the belief graph based on a market event.
        
        This is the core function that ingests market events and revises
        the belief graph accordingly.
        """
        self.current_time = event.timestamp
        self.event_history.append(event)
        
        # Ensure the agent exists in the graph
        if event.agent_id and event.agent_id not in self.nodes:
            self.add_agent(event.agent_id)
        
        # Update based on event type
        if event.event_type == EventType.BID:
            self._update_beliefs_from_bid(event)
        elif event.event_type == EventType.ASK:
            self._update_beliefs_from_ask(event)
        elif event.event_type == EventType.TRADE:
            self._update_beliefs_from_trade(event)
        elif event.event_type == EventType.CANCEL:
            self._update_beliefs_from_cancel(event)
        
        # Update asset state
        self._update_asset_state()
        
        # Decay old beliefs
        self._decay_old_beliefs()
    
    def _update_beliefs_from_bid(self, event: MarketEvent) -> None:
        """Update beliefs based on a bid event"""
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_bid_price = event.price
        agent_node.last_activity = event.timestamp
        
        # Update discrete beliefs using set elimination logic
        self._update_discrete_beliefs_from_market_event(event.agent_id, event.price, "bid")
    
    def _update_beliefs_from_ask(self, event: MarketEvent) -> None:
        """Update beliefs based on an ask event"""
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_ask_price = event.price
        agent_node.last_activity = event.timestamp
        
        # Update valuation belief
        self._update_valuation_belief(event.agent_id, event.price, "ask")
    
    def _update_beliefs_from_trade(self, event: MarketEvent) -> None:
        """Update beliefs based on a trade event"""
        gv1_logger.debug(f"[BG-DEBUG] _update_beliefs_from_trade called for agent {event.agent_id} at price {event.price}")
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        old_aggr = agent_node.aggressiveness_score
        agent_node.last_trade_price = event.price
        agent_node.total_trades += 1
        agent_node.total_volume += event.quantity or 1
        agent_node.last_activity = event.timestamp
        
        # Update aggressiveness based on trade price compared to previous trades FIRST
        # (so desperation updates can use the aggressiveness score)
        if self.asset_node.last_trade_price and agent_node.total_trades > 1:
            price_ratio = event.price / self.asset_node.last_trade_price
            gv1_logger.debug(f"[BG-TRADE-AGGR] Agent {event.agent_id}: price={event.price}, last_price={self.asset_node.last_trade_price}, ratio={price_ratio:.3f}")
            
            # Buyer perspective (assuming agent_id is buyer in BSE)
            if price_ratio > 1.01:  # Paid >1% more than last trade
                agent_node.aggressiveness_score = min(1.0, agent_node.aggressiveness_score + 0.3)
                gv1_logger.debug(f"[BG-TRADE-AGGR] AGGRESSIVE buyer: paid {price_ratio-1:.1%} more")
            elif price_ratio < 0.99:  # Paid <1% less than last trade  
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.2)
                gv1_logger.debug(f"[BG-TRADE-AGGR] PASSIVE buyer: paid {1-price_ratio:.1%} less")
            else:
                # Near market price - slight shift toward passive
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.05)
                gv1_logger.debug(f"[BG-TRADE-AGGR] NEUTRAL trade")
        else:
            # First trade - random initial aggressiveness
            import random
            agent_node.aggressiveness_score = random.uniform(-0.3, 0.3)
            gv1_logger.debug(f"[BG-TRADE-AGGR] First trade, random initial aggr={agent_node.aggressiveness_score:.3f}")
        
        gv1_logger.debug(f"[BG-TRADE-AGGR] Agent {event.agent_id} aggr: {old_aggr:.3f} -> {agent_node.aggressiveness_score:.3f}")
        
        # Update valuation belief with high confidence (actual trade)
        self._update_valuation_belief(event.agent_id, event.price, "trade", high_confidence=True)
        # Update ALL other GraphVar1 belief types based on trade behavior
        gv1_logger.debug(f"[GraphVar1-CALLING] About to call belief update methods for {event.agent_id}")
        self._update_direction_belief_discrete(event.agent_id, event.price, "trade")
        self._update_desperation_belief_discrete(event.agent_id, event.price, "trade")
        self._update_cash_belief_discrete(event.agent_id, event.price, "trade")
        self._update_exit_strategy_belief_discrete(event.agent_id, event.price, "trade")
        gv1_logger.debug(f"[GraphVar1-CALLED] Finished calling belief update methods for {event.agent_id}")
        
        # Update strategy belief based on trade
        self._update_strategy_belief(event.agent_id, "trade", event.price)
        
        # Also update counterparty if available
        if event.counterparty_id and event.counterparty_id in self.nodes:
            seller_node = self.nodes[event.counterparty_id]
            old_seller_aggr = seller_node.aggressiveness_score
            seller_node.last_trade_price = event.price
            seller_node.total_trades += 1
            seller_node.total_volume += event.quantity or 1
            
            # Seller perspective - opposite of buyer
            if self.asset_node.last_trade_price and seller_node.total_trades > 1:
                price_ratio = event.price / self.asset_node.last_trade_price
                
                if price_ratio < 0.99:  # Sold <1% below last trade
                    seller_node.aggressiveness_score = min(1.0, seller_node.aggressiveness_score + 0.3)
                    gv1_logger.debug(f"[BG-TRADE-AGGR] AGGRESSIVE seller {event.counterparty_id}: sold {1-price_ratio:.1%} below")
                elif price_ratio > 1.01:  # Sold >1% above last trade
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.2)
                    gv1_logger.debug(f"[BG-TRADE-AGGR] PASSIVE seller {event.counterparty_id}: sold {price_ratio-1:.1%} above")
                else:
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.05)
            else:
                # First trade - random initial
                import random
                seller_node.aggressiveness_score = random.uniform(-0.3, 0.3)
                
            gv1_logger.debug(f"[BG-TRADE-AGGR] Seller {event.counterparty_id} aggr: {old_seller_aggr:.3f} -> {seller_node.aggressiveness_score:.3f}")
            
            # Update seller strategy
            self._update_valuation_belief(event.counterparty_id, event.price, "trade", high_confidence=True)
            self._update_strategy_belief(event.counterparty_id, "trade", event.price)
        
        # Update asset state
        self.asset_node.last_trade_price = event.price
        self.asset_node.volume_traded += event.quantity or 1
    
    def _update_beliefs_from_cancel(self, event: MarketEvent) -> None:
        """Update beliefs based on a cancel event"""
        if not event.agent_id:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_activity = event.timestamp
        
        # Cancellation might indicate uncertainty or strategy change
        agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.05)
    
    def _update_valuation_belief(self, agent_id: str, price: float, action_type: str, high_confidence: bool = False) -> None:
        """Update discrete valuation beliefs using set elimination logic"""
        # Find existing valuation edge
        valuation_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "valuation"):
                valuation_edge = edge
                break
        
        if valuation_edge is None:
            # Create new valuation edge with discrete set
            valuation_edge = BeliefEdge(
                edge_id=str(uuid.uuid4()),
                source_node=self.asset_id,
                target_node=agent_id,
                belief_type="valuation",
                confidence=1.0,  # Full confidence in the discrete set
                value={"possible_valuations": [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]},
                timestamp=self.current_time
            )
            self.edges[valuation_edge.edge_id] = valuation_edge
        
        # Apply discrete set elimination based on market action
        if valuation_edge.value and isinstance(valuation_edge.value, dict) and "possible_valuations" in valuation_edge.value:
            possible_valuations = valuation_edge.value["possible_valuations"][:]
            
            # Apply Hanabi-style elimination logic
            if action_type == "bid":
                # If agent bids at price P, their valuation is likely >= P
                # Remove valuations significantly below the bid price
                elimination_threshold = price - 5  # Allow some buffer
                possible_valuations = [val for val in possible_valuations if val >= elimination_threshold]
            
            elif action_type == "ask":
                # If agent asks at price P, their valuation is likely <= P  
                # Remove valuations significantly above the ask price
                elimination_threshold = price + 5  # Allow some buffer
                possible_valuations = [val for val in possible_valuations if val <= elimination_threshold]
            
            elif action_type == "trade":
                # Trade provides strongest signal - agent's valuation is very close to trade price
                # Keep only valuations within tight range of trade price
                lower_bound = price - 10
                upper_bound = price + 10
                possible_valuations = [val for val in possible_valuations if lower_bound <= val <= upper_bound]
            
            # If the price is not in our possibilities, add it
            if int(price) not in possible_valuations and int(price) not in valuation_edge.value["possible_valuations"]:
                # Add the new price as a possibility
                possible_valuations.append(int(price))
                possible_valuations.sort()
                gv1_logger.info(f"[GraphVar1-NEW-PRICE] Added {int(price)} to possible valuations for {agent_id}")
            
            # Ensure we don't eliminate all possibilities
            if not possible_valuations:
                # Fallback - keep original set but add the observed price as evidence
                possible_valuations = valuation_edge.value["possible_valuations"][:]
                if int(price) not in possible_valuations:
                    # Add the new price as a possibility
                    possible_valuations.append(int(price))
                    possible_valuations.sort()
            
            # Log before/after state
            old_valuations = valuation_edge.value.get("possible_valuations", []) if valuation_edge.value else []
            gv1_logger.debug(f"[GraphVar1-VALUATION-BEFORE] {agent_id}: {old_valuations}")
            
            # Update the edge with new discrete set
            valuation_edge.value = {"possible_valuations": possible_valuations}
            valuation_edge.timestamp = self.current_time
            valuation_edge.evidence_count += 1
            
            gv1_logger.debug(f"[GraphVar1-VALUATION-AFTER] {agent_id}: {possible_valuations}")
            gv1_logger.debug(f"[GraphVar1-VALUATION-TRIGGER] Action={action_type}, Price={price}, Evidence Count={valuation_edge.evidence_count}")
            
            # Sync the agent node's valuation fields with the discrete set
            agent_node = self.nodes[agent_id]
            # Use the middle value of the remaining possibilities as a single estimate
            if possible_valuations:
                agent_node.inferred_valuation = possible_valuations[len(possible_valuations)//2]
                agent_node.valuation_confidence = 1.0 - (len(possible_valuations) / 27.0)  # Higher confidence with fewer possibilities (27 initial values)

    
    def _update_strategy_belief(self, agent_id: str, action_type: str, price: float) -> None:
        """Update the belief about an agent's strategy"""
        gv1_logger.debug(f"[BG-STRAT-START] Updating strategy for {agent_id}, action={action_type}, price={price}")
        
        # Find existing strategy edge
        strategy_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "strategy"):
                strategy_edge = edge
                break
        
        if strategy_edge is None:
            gv1_logger.debug(f"[BG-STRAT-ERROR] No strategy edge found for {agent_id}")
            return
        
        # Simple strategy classification based on behavior patterns
        agent_node = self.nodes[agent_id]
        
        gv1_logger.debug(f"[BG-STRAT-INFO] Agent {agent_id}: trades={agent_node.total_trades}, aggr_score={agent_node.aggressiveness_score:.3f}, current_strategy={agent_node.strategy_type}")
        
        # Always classify based on aggressiveness, even if no trades yet
        old_strategy = agent_node.strategy_type
        
        # Use more sensitive thresholds
        if agent_node.aggressiveness_score > 0.15:
            strategy = "aggressive"
        elif agent_node.aggressiveness_score < -0.15:
            strategy = "passive"
        else:
            strategy = "neutral"
        
        gv1_logger.debug(f"[BG-STRAT-CLASSIFY] Agent {agent_id}: aggr={agent_node.aggressiveness_score:.3f} -> strategy={strategy} (was {old_strategy})")
        
        strategy_edge.value = strategy
        strategy_edge.confidence = min(self.max_confidence, strategy_edge.confidence + 0.1)
        strategy_edge.timestamp = self.current_time
        strategy_edge.evidence_count += 1
        
        # Sync the agent node's strategy_type field
        agent_node.strategy_type = strategy
        
        gv1_logger.debug(f"[BG-STRAT-END] Agent {agent_id} strategy updated to {strategy} (confidence={strategy_edge.confidence:.2f})")
    
    def _update_asset_state(self) -> None:
        """Update the asset node state based on current market conditions"""
        # This would typically be called with actual market data
        # For now, we'll update based on the belief graph state
        
        # Calculate spread if we have both bid and ask
        if (self.asset_node.current_best_bid is not None and 
            self.asset_node.current_best_ask is not None):
            self.asset_node.spread_width = self.asset_node.current_best_ask - self.asset_node.current_best_bid
    
    def _decay_old_beliefs(self) -> None:
        """Decay confidence in old beliefs"""
        current_time = self.current_time
        for edge in self.edges.values():
            time_diff = current_time - edge.timestamp
            if time_diff > 100:  # Decay beliefs older than 100 time units
                decay_factor = self.valuation_decay_rate ** (time_diff / 100)
                edge.confidence *= decay_factor
    
    
    def _update_self_beliefs(self, agent_id: str, market_state: Dict[str, Any]) -> None:
        """Update agent's beliefs about their own market assessment and strategy"""
        # Find self-belief edges for this agent
        self_direction_edge = None
        self_price_edge = None
        self_urgency_edge = None
        
        for edge in self.edges.values():
            if (edge.source_node == agent_id and edge.target_node == agent_id):
                if edge.belief_type == "self_market_direction":
                    self_direction_edge = edge
                elif edge.belief_type == "self_optimal_entry":
                    self_price_edge = edge
                elif edge.belief_type == "self_time_urgency":
                    self_urgency_edge = edge
        
        # Update market direction belief based on recent price movements
        if self_direction_edge and self_direction_edge.value:
            current_directions = self_direction_edge.value.get("possible_directions", ["up", "down", "sideways"])
            
            # Analyze recent price trend
            if 'best_bid' in market_state and 'best_ask' in market_state:
                bid, ask = market_state['best_bid'], market_state['best_ask']
                if bid and ask:
                    spread = ask - bid
                    # Wide spread might indicate uncertainty -> keep all directions
                    # Narrow spread might indicate consensus -> eliminate less likely directions
                    if spread <= 5:  # Narrow spread suggests direction consensus
                        if len(current_directions) > 1:
                            # Keep fewer directions (eliminate one randomly based on market bias)
                            if "sideways" in current_directions and len(current_directions) > 2:
                                current_directions.remove("sideways")
                    
            self_direction_edge.value = {"possible_directions": current_directions}
            self_direction_edge.timestamp = self.current_time
        
        # Update price belief based on current market conditions
        if self_price_edge and self_price_edge.value:
            current_prices = self_price_edge.value.get("possible_prices", [85, 90, 95, 100, 105, 110, 115])
            
            # Narrow price range based on observed market activity
            if 'best_bid' in market_state and 'best_ask' in market_state:
                bid, ask = market_state['best_bid'], market_state['best_ask']
                if bid and ask:
                    # Focus on prices within reasonable range of current market
                    market_mid = (bid + ask) / 2
                    # Keep prices within ±15 of market midpoint
                    current_prices = [p for p in current_prices if abs(p - market_mid) <= 15]
                    
            self_price_edge.value = {"possible_prices": current_prices}
            self_price_edge.timestamp = self.current_time
        
        # Update urgency belief based on time remaining
        if self_urgency_edge and self_urgency_edge.value:
            current_urgency = self_urgency_edge.value.get("possible_urgency", ["low", "medium", "high"])
            
            # Increase urgency as time passes
            time_remaining = market_state.get('time_remaining', 200)
            if time_remaining < 50:  # Less than 50 time units left
                # Eliminate low urgency
                if "low" in current_urgency:
                    current_urgency.remove("low")
            elif time_remaining < 100:  # Less than 100 time units left
                # Add medium urgency if not present
                if "medium" not in current_urgency:
                    current_urgency.append("medium")
            
            self_urgency_edge.value = {"possible_urgency": current_urgency}
            self_urgency_edge.timestamp = self.current_time

    def _get_dynamic_self_beliefs(self, agent_id: str, current_market_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get agent's self-beliefs from stored edges"""
        # Find self-belief edges for this agent
        self_direction_edge = None
        self_price_edge = None
        self_urgency_edge = None
        
        for edge in self.edges.values():
            if (edge.source_node == agent_id and edge.target_node == agent_id):
                if edge.belief_type == "self_market_direction":
                    self_direction_edge = edge
                elif edge.belief_type == "self_optimal_entry":
                    self_price_edge = edge
                elif edge.belief_type == "self_time_urgency":
                    self_urgency_edge = edge
        
        # Generate actual situation descriptions based on market state
        actual_signals = "mixed_price_movements"
        if 'best_bid' in current_market_state and 'best_ask' in current_market_state:
            bid, ask = current_market_state.get('best_bid'), current_market_state.get('best_ask')
            if bid and ask:
                spread = ask - bid
                if spread > 15:
                    actual_signals = "wide_spread_uncertainty"
                elif spread < 5:
                    actual_signals = "tight_spread_consensus"
                else:
                    actual_signals = "moderate_spread_activity"
        
        actual_situation = "volatile_spreads"
        if 'best_bid' in current_market_state and 'best_ask' in current_market_state:
            bid, ask = current_market_state.get('best_bid'), current_market_state.get('best_ask')
            if bid and ask:
                market_mid = (bid + ask) / 2
                if market_mid < 95:
                    actual_situation = "low_market_prices"
                elif market_mid > 105:
                    actual_situation = "high_market_prices"
                else:
                    actual_situation = "balanced_market_prices"
        
        actual_urgency_situation = "time_pressure_building"
        time_remaining = current_market_state.get('time_remaining', 200)
        if time_remaining < 50:
            actual_urgency_situation = "urgent_time_pressure"
        elif time_remaining < 100:
            actual_urgency_situation = "moderate_time_pressure"
        else:
            actual_urgency_situation = "comfortable_time_remaining"
        
        return {
            "My_Market_Direction": {
                "actual_signals_I_see": actual_signals,
                "my_self_belief": {
                    "possible_directions": self_direction_edge.value.get("possible_directions", ["up", "down", "sideways"]) if self_direction_edge and self_direction_edge.value else ["up", "down", "sideways"]
                }
            },
            "My_Optimal_Entry_Price": {
                "actual_situation_I_face": actual_situation,
                "my_self_belief": {
                    "possible_prices": self_price_edge.value.get("possible_prices", [85, 90, 95, 100, 105, 110, 115]) if self_price_edge and self_price_edge.value else [85, 90, 95, 100, 105, 110, 115]
                }
            },
            "My_Time_Urgency": {
                "actual_situation_I_face": actual_urgency_situation,
                "my_self_belief": {
                    "possible_urgency": self_urgency_edge.value.get("possible_urgency", ["low", "medium", "high"]) if self_urgency_edge and self_urgency_edge.value else ["low", "medium", "high"]
                }
            }
        }
    
    def query_action(self, agent_id: str, current_market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the belief graph for decision-making.
        
        This function serializes the belief graph and returns it in a format
        suitable for LLM processing.
        """
        # Update asset state with current market data
        if 'best_bid' in current_market_state:
            self.asset_node.current_best_bid = current_market_state['best_bid']
        if 'best_ask' in current_market_state:
            self.asset_node.current_best_ask = current_market_state['best_ask']
        if 'last_trade' in current_market_state:
            self.asset_node.last_trade_price = current_market_state['last_trade']
        
        # Update self-beliefs based on current market conditions
        self._update_self_beliefs(agent_id, current_market_state)
        
        # Generate competitor beliefs using existing edges
        # Generate competitor beliefs using actual stored edges
        competitor_beliefs = {}
        for node_id, node in self.nodes.items():
            if isinstance(node, AgentNode) and node_id != agent_id:
                # Observe behavior with rich context
                observed_behavior = "passive_observation"
                if node.last_bid_price:
                    # Generate contextual bid behavior
                    current_ask = self.asset_node.current_best_ask
                    if current_ask and node.last_bid_price >= current_ask * 0.95:
                        observed_behavior = f"aggressive_bidding_at_{node.last_bid_price}"
                    elif node.aggressiveness_score > 0.1:
                        observed_behavior = f"active_bidding_at_{node.last_bid_price}"
                    else:
                        observed_behavior = f"cautious_bidding_at_{node.last_bid_price}"
                elif node.last_ask_price:
                    # Generate contextual ask behavior  
                    current_bid = self.asset_node.current_best_bid
                    if current_bid and node.last_ask_price <= current_bid * 1.05:
                        observed_behavior = f"aggressive_asking_at_{node.last_ask_price}"
                    elif node.aggressiveness_score > 0.1:
                        observed_behavior = f"active_asking_at_{node.last_ask_price}"
                    else:
                        observed_behavior = f"cautious_asking_at_{node.last_ask_price}"
                elif node.last_trade_price:
                    # Trading behavior context
                    if node.total_trades >= 3:
                        observed_behavior = f"frequent_trading_at_{node.last_trade_price}"
                    elif node.aggressiveness_score > 0.2:
                        observed_behavior = f"aggressive_trading_at_{node.last_trade_price}"
                    else:
                        observed_behavior = f"careful_trading_at_{node.last_trade_price}"
                
                competitor_beliefs[f"{node_id}_Beliefs"] = {}
                
                # Get all belief types for this competitor from stored edges
                for edge in self.edges.values():
                    if edge.target_node == node_id and edge.value and isinstance(edge.value, dict):
                        
                        if edge.belief_type == "valuation" and "possible_valuations" in edge.value:
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Valuation"] = {
                                "actual_behavior_I_observe": observed_behavior,
                                f"{node_id.lower()}_self_belief": {
                                    "possible_valuations": edge.value["possible_valuations"]
                                }
                            }
                        
                        elif edge.belief_type == "market_direction" and "possible_directions" in edge.value:
                            # Rich timing behavior context
                            if node.aggressiveness_score <= -0.1:
                                timing_behavior = "very_cautious_timing"
                            elif node.aggressiveness_score <= 0.1:
                                timing_behavior = "cautious_timing"
                            elif node.aggressiveness_score <= 0.3:
                                timing_behavior = "moderate_timing"
                            elif node.aggressiveness_score <= 0.6:
                                timing_behavior = "urgent_timing"
                            else:
                                timing_behavior = "extremely_urgent_timing"
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Market_Direction"] = {
                                "actual_behavior_I_observe": timing_behavior,
                                f"{node_id.lower()}_self_belief": {
                                    "possible_directions": edge.value["possible_directions"]
                                }
                            }
                        
                        elif edge.belief_type == "desperation_level" and "possible_desperation" in edge.value:
                            # Rich desperation behavior context
                            if node.aggressiveness_score <= -0.2:
                                desperation_behavior = "extremely_patient_small_bids"
                            elif node.aggressiveness_score <= 0.0:
                                desperation_behavior = "patient_small_bids"
                            elif node.aggressiveness_score <= 0.2:
                                desperation_behavior = "moderate_sized_orders"
                            elif node.aggressiveness_score <= 0.4:
                                desperation_behavior = "moderate_pressure_bids"
                            elif node.aggressiveness_score <= 0.7:
                                desperation_behavior = "urgent_larger_orders"
                            else:
                                desperation_behavior = "large_urgent_orders"
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Desperation_Level"] = {
                                "actual_behavior_I_observe": desperation_behavior,
                                f"{node_id.lower()}_self_belief": {
                                    "possible_desperation": edge.value["possible_desperation"]
                                }
                            }
                        
                        elif edge.belief_type == "available_cash" and "possible_cash" in edge.value:
                            # Rich cash behavior context
                            if node.total_volume <= 1:
                                cash_behavior = "very_small_consistent_volumes"
                            elif node.total_volume <= 3:
                                cash_behavior = "consistent_small_volumes"
                            elif node.total_volume <= 6:
                                cash_behavior = "moderate_position_sizes"
                            elif node.total_volume <= 10:
                                cash_behavior = "larger_position_sizes"
                            else:
                                cash_behavior = "very_large_position_sizes"
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Available_Cash"] = {
                                "actual_behavior_I_observe": cash_behavior,
                                f"{node_id.lower()}_self_belief": {
                                    "possible_cash": edge.value["possible_cash"]
                                }
                            }
                        
                        elif edge.belief_type == "exit_strategy" and "possible_exits" in edge.value:
                            # Rich exit strategy behavior context
                            if node.total_trades == 0:
                                exit_behavior = "no_trading_activity"
                            elif node.total_trades == 1:
                                exit_behavior = "minimal_trading"
                            elif node.total_trades <= 2:
                                exit_behavior = "holding_positions"
                            elif node.total_trades <= 4:
                                exit_behavior = "moderate_trading"
                            elif node.total_trades <= 7:
                                exit_behavior = "active_trading"
                            else:
                                exit_behavior = "very_active_trading"
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Exit_Strategy"] = {
                                "actual_behavior_I_observe": exit_behavior,
                                f"{node_id.lower()}_self_belief": {
                                    "possible_exits": edge.value["possible_exits"]
                                }
                            }
        
        # GraphVar1 returns pure discrete structures
        return {
            "MarketState": {
                "current_bid": current_market_state.get('best_bid'),
                "current_ask": current_market_state.get('best_ask'), 
                "time_remaining": current_market_state.get('time_remaining', 0)
            },
            "My_Trading_Beliefs": self._get_dynamic_self_beliefs(agent_id, current_market_state),
            "Competitor_Trading_Beliefs": competitor_beliefs
        }
    
    def _generate_strategic_insights(self, agent_id: str) -> Dict[str, Any]:
        """Generate strategic insights for the querying agent"""
        insights = {
            'competitors': [],
            'market_opportunities': [],
            'risk_factors': []
        }
        
        # Analyze competitors
        for node_id, node in self.nodes.items():
            if isinstance(node, AgentNode) and node_id != agent_id:
                competitor_info = {
                    'agent_id': node_id,
                    'strategy': node.strategy_type or "unknown",
                    'aggressiveness': node.aggressiveness_score,
                    'valuation_estimate': node.inferred_valuation,
                    'confidence': node.valuation_confidence,
                    'recent_activity': node.last_activity
                }
                insights['competitors'].append(competitor_info)
        
        # Identify market opportunities
        if (self.asset_node.current_best_bid is not None and 
            self.asset_node.current_best_ask is not None):
            spread = self.asset_node.current_best_ask - self.asset_node.current_best_bid
            if spread > 5:  # Arbitrage opportunity
                insights['market_opportunities'].append({
                    'type': 'arbitrage',
                    'spread': spread,
                    'description': f"Large spread of {spread} points"
                })
        
        # Identify risk factors
        if self.asset_node.price_volatility > 0.1:
            insights['risk_factors'].append({
                'type': 'high_volatility',
                'value': self.asset_node.price_volatility,
                'description': "High price volatility detected"
            })
        
        return insights
    
    def to_json(self) -> str:
        """Serialize the belief graph to JSON"""
        graph_data = {
            'graph_id': self.graph_id,
            'asset_id': self.asset_id,
            'current_time': self.current_time,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            'event_history': [event.to_dict() for event in self.event_history[-50:]]  # Last 50 events
        }
        return json.dumps(graph_data, indent=2)
    
    def from_json(self, json_str: str) -> None:
        """Deserialize the belief graph from JSON"""
        data = json.loads(json_str)
        self.graph_id = data['graph_id']
        self.asset_id = data['asset_id']
        self.current_time = data['current_time']
        
        # Reconstruct nodes
        self.nodes.clear()
        for node_id, node_data in data['nodes'].items():
            if node_data['node_type'] == NodeType.AGENT.value:
                self.nodes[node_id] = AgentNode(**node_data)
            elif node_data['node_type'] == NodeType.ASSET.value:
                self.nodes[node_id] = AssetNode(**node_data)
        
        # Reconstruct edges
        self.edges.clear()
        for edge_id, edge_data in data['edges'].items():
            self.edges[edge_id] = BeliefEdge(**edge_data)
        
        # Reconstruct event history
        self.event_history = [MarketEvent(**event_data) for event_data in data['event_history']]
    
    def get_agent_beliefs(self, agent_id: str) -> Dict[str, Any]:
        """Get all beliefs about a specific agent"""
        beliefs = {}
        
        # First get the raw belief edges
        for edge in self.edges.values():
            if edge.target_node == agent_id:
                if edge.belief_type not in beliefs:
                    beliefs[edge.belief_type] = []
                beliefs[edge.belief_type].append(edge.to_dict())
        
        # Extract valuation estimate if available
        valuation_estimate = None
        if agent_id in self.nodes:
            agent_node = self.nodes[agent_id]
            if hasattr(agent_node, 'inferred_valuation'):
                valuation_estimate = agent_node.inferred_valuation
            
            # Also try to get from valuation edge
            for edge in self.edges.values():
                if (edge.target_node == agent_id and 
                    edge.belief_type == "valuation" and 
                    edge.value and isinstance(edge.value, dict)):
                    possible_vals = edge.value.get("possible_valuations", [])
                    if possible_vals:
                        # Use median of possible valuations as estimate
                        valuation_estimate = sorted(possible_vals)[len(possible_vals)//2]
                        break
        
        # Extract strategy type if available
        strategy_type = None
        if agent_id in self.nodes:
            agent_node = self.nodes[agent_id]
            if hasattr(agent_node, 'strategy_type'):
                strategy_type = agent_node.strategy_type
        
        # Add simplified summary
        beliefs['valuation_estimate'] = valuation_estimate
        beliefs['strategy_type'] = strategy_type
        
        return beliefs
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get a summary of the current market state"""
        return {
            'asset_state': self.asset_node.to_dict(),
            'active_agents': len([n for n in self.nodes.values() if isinstance(n, AgentNode)]),
            'total_beliefs': len(self.edges),
            'recent_events': len(self.event_history),
            'current_time': self.current_time
        }

    def _update_direction_belief_discrete(self, agent_id: str, price: float, action_type: str) -> None:
        """Update discrete market direction beliefs based on trade behavior"""
        gv1_logger.debug(f"[GraphVar1-DIRECTION-START] Updating direction belief for {agent_id}, price={price}, action={action_type}")
        direction_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "market_direction"):
                direction_edge = edge
                break
        
        if direction_edge and "possible_directions" in direction_edge.value:
            gv1_logger.debug(f"[GraphVar1-DIRECTION-FOUND] Found direction edge for {agent_id}")
            current_set = set(direction_edge.value["possible_directions"])
            old_set = current_set.copy()
            
            # Update based on price trend (discrete elimination)
            if self.asset_node.last_trade_price:
                if price > self.asset_node.last_trade_price * 1.02:  # Price going up
                    current_set.discard("down")  # Remove "down" belief
                elif price < self.asset_node.last_trade_price * 0.98:  # Price going down  
                    current_set.discard("up")  # Remove "up" belief
                else:  # Sideways movement
                    # Keep all options for sideways
                    pass
                        
                # Log changes
                if current_set != old_set:
                    removed = old_set - current_set
                    gv1_logger.info(f"[GraphVar1-DIRECTION-UPDATE] {agent_id} {action_type.upper()} at {price}: eliminated {removed}")
                
                direction_edge.value["possible_directions"] = list(current_set)
                direction_edge.timestamp = self.current_time
                direction_edge.evidence_count += 1
                gv1_logger.debug(f"[GraphVar1-DIRECTION-COMPLETE] Updated direction belief for {agent_id}")
        else:
            gv1_logger.debug(f"[GraphVar1-DIRECTION-NO-EDGE] No direction edge found for {agent_id}")
    
    def _update_desperation_belief_discrete(self, agent_id: str, price: float, action_type: str) -> None:
        """Update discrete desperation beliefs based on trading behavior"""
        gv1_logger.debug(f"[GraphVar1-DESPERATION-START] Updating desperation belief for {agent_id}, price={price}, action={action_type}")
        desperation_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "desperation_level"):
                desperation_edge = edge
                break
        
        if desperation_edge and "possible_desperation" in desperation_edge.value:
            current_set = set(desperation_edge.value["possible_desperation"])
            old_set = current_set.copy()
            
            # Analyze trading behavior for desperation signals
            agent_node = self.nodes[agent_id]
            gv1_logger.debug(f"[GraphVar1-DESPERATION-AGGR] {agent_id} aggressiveness_score={agent_node.aggressiveness_score}")
            
            # High aggressiveness suggests not calm
            if agent_node.aggressiveness_score > 0.1:  # Lowered threshold
                current_set.discard("calm")
                gv1_logger.debug(f"[GraphVar1-DESPERATION-LOGIC] {agent_id} aggressive, removing calm")
            elif agent_node.aggressiveness_score < -0.1:  # Lowered threshold
                current_set.discard("desperate")
                gv1_logger.debug(f"[GraphVar1-DESPERATION-LOGIC] {agent_id} passive, removing desperate")
            else:
                # Moderate aggressiveness
                gv1_logger.debug(f"[GraphVar1-DESPERATION-LOGIC] {agent_id} moderate, no elimination")
                
            # Log changes
            if current_set != old_set:
                removed = old_set - current_set
                gv1_logger.info(f"[GraphVar1-DESPERATION-UPDATE] {agent_id} {action_type.upper()} at {price}: eliminated {removed}")
            
            desperation_edge.value["possible_desperation"] = list(current_set)
            desperation_edge.timestamp = self.current_time
            desperation_edge.evidence_count += 1
            gv1_logger.debug(f"[GraphVar1-DESPERATION-COMPLETE] Updated desperation belief for {agent_id}")
        else:
            gv1_logger.debug(f"[GraphVar1-DESPERATION-NO-EDGE] No desperation edge found for {agent_id}")
    
    def _update_cash_belief_discrete(self, agent_id: str, price: float, action_type: str) -> None:
        """Update discrete cash beliefs based on trading volume"""
        gv1_logger.debug(f"[GraphVar1-CASH-START] Updating cash belief for {agent_id}, price={price}, action={action_type}")
        cash_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "available_cash"):
                cash_edge = edge
                break
        
        if cash_edge and "possible_cash" in cash_edge.value:
            current_set = set(cash_edge.value["possible_cash"])
            old_set = current_set.copy()
            
            # Analyze volume for cash inference
            agent_node = self.nodes[agent_id]
            
            # High volume suggests not low cash
            if agent_node.total_volume >= 3:
                current_set.discard("low")
            elif agent_node.total_volume >= 2:
                # Medium volume, keep all options
                pass
            else:
                # Low volume suggests not high cash
                current_set.discard("high")
                
            # Log changes
            if current_set != old_set:
                removed = old_set - current_set
                gv1_logger.info(f"[GraphVar1-CASH-UPDATE] {agent_id} {action_type.upper()} at {price} (volume={agent_node.total_volume}): eliminated {removed}")
            
            cash_edge.value["possible_cash"] = list(current_set)
            cash_edge.timestamp = self.current_time
            cash_edge.evidence_count += 1
    
    def _update_exit_strategy_belief_discrete(self, agent_id: str, price: float, action_type: str) -> None:
        """Update discrete exit strategy beliefs based on trading patterns"""
        exit_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "exit_strategy"):
                exit_edge = edge
                break
        
        if exit_edge and "possible_exits" in exit_edge.value:
            current_set = set(exit_edge.value["possible_exits"])
            old_set = current_set.copy()
            
            # Analyze trading frequency for exit strategy
            agent_node = self.nodes[agent_id]
            
            # Frequent trading suggests not hold_till_end
            if agent_node.total_trades >= 3:
                current_set.discard("hold_till_end")
            elif agent_node.total_trades >= 2:
                # Medium trading, keep all options
                pass
            else:
                # Few trades suggests not opportunistic
                current_set.discard("opportunistic")
                
            # Log changes
            if current_set != old_set:
                removed = old_set - current_set
                gv1_logger.info(f"[GraphVar1-EXIT-UPDATE] {agent_id} {action_type.upper()} at {price} (trades={agent_node.total_trades}): eliminated {removed}")
            
            exit_edge.value["possible_exits"] = list(current_set)
            exit_edge.timestamp = self.current_time
            exit_edge.evidence_count += 1


class GraphVar2:
    """
    Main belief graph class for managing agent beliefs and market state using probabilistic distributions.
    
    The belief graph maintains:
    - Nodes for each agent and the traded asset
    - Edges representing probability distributions about other agents' valuations and strategies
    - Bayesian updates based on market events
    - Query interface for decision-making
    """
    
    def __init__(self, asset_id: str = "DEFAULT_ASSET"):
        self.graph_id = str(uuid.uuid4())
        self.asset_id = asset_id
        self.nodes: Dict[str, AgentNode | AssetNode] = {}
        self.edges: Dict[str, BeliefEdge] = {}
        self.event_history: List[MarketEvent] = []
        self.current_time = 0.0
        
        # Initialize the asset node
        self.asset_node = AssetNode(asset_id=asset_id)
        self.nodes[asset_id] = self.asset_node
        
        # Belief update parameters - CONFIGURABLE!
        self.valuation_decay_rate = 0.95  # How quickly old valuation beliefs decay
        self.confidence_boost = 0.1  # How much confidence increases with new evidence
        self.max_confidence = 0.95  # Maximum confidence level
        
        # GraphVar2 Bayesian update parameters - configurable likelihoods
        self.bid_support_likelihood = 0.8  # P(bid at price | valuation >= price)
        self.bid_contradict_likelihood = 0.2  # P(bid at price | valuation < price)
        self.ask_support_likelihood = 0.8  # P(ask at price | valuation <= price)
        self.ask_contradict_likelihood = 0.2  # P(ask at price | valuation > price)
        self.trade_close_likelihood = 0.9  # P(trade at price | valuation close to price)
        self.trade_medium_likelihood = 0.6  # P(trade at price | valuation medium distance)
        self.trade_far_likelihood = 0.1  # P(trade at price | valuation far from price)
        self.trade_close_distance = 5  # Distance threshold for "close" to trade price
        self.trade_medium_distance = 10  # Distance threshold for "medium" from trade price
        
    def add_agent(self, agent_id: str) -> None:
        """Add a new agent to the belief graph"""
        if agent_id not in self.nodes:
            agent_node = AgentNode(agent_id=agent_id)
            self.nodes[agent_id] = agent_node
            
            # Add initial beliefs about this agent
            self._add_initial_beliefs(agent_id)
    
    def _add_initial_beliefs(self, agent_id: str) -> None:
        """Add initial probability distributions about a new agent"""
        # Add probabilistic belief about agent's valuation  
        valuation_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="valuation",
            confidence=1.0,  # Full confidence in the distribution
            value={"valuation_distribution": {"70": 0.037, "75": 0.037, "80": 0.037, "85": 0.037, "90": 0.037, "95": 0.037, "100": 0.037, "105": 0.037, "110": 0.037, "115": 0.037, "120": 0.037, "125": 0.037, "130": 0.037, "135": 0.037, "140": 0.037, "145": 0.037, "150": 0.037, "155": 0.037, "160": 0.037, "165": 0.037, "170": 0.037, "175": 0.037, "180": 0.037, "185": 0.037, "190": 0.037, "195": 0.037, "200": 0.038}},
            timestamp=self.current_time
        )
        self.edges[valuation_edge.edge_id] = valuation_edge
        
        # Add probabilistic belief about agent's market direction
        direction_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="market_direction",
            confidence=1.0,
            value={"direction_distribution": {"up": 0.33, "down": 0.33, "sideways": 0.34}},
            timestamp=self.current_time
        )
        self.edges[direction_edge.edge_id] = direction_edge
        
        # Add probabilistic belief about agent's desperation level
        desperation_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="desperation_level",
            confidence=1.0,
            value={"desperation_distribution": {"calm": 0.4, "moderate": 0.4, "desperate": 0.2}},
            timestamp=self.current_time
        )
        self.edges[desperation_edge.edge_id] = desperation_edge
        
        # Add probabilistic belief about agent's available cash
        cash_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="available_cash",
            confidence=1.0,
            value={"cash_distribution": {"low": 0.3, "medium": 0.4, "high": 0.3}},
            timestamp=self.current_time
        )
        self.edges[cash_edge.edge_id] = cash_edge
        
        # Add probabilistic belief about agent's exit strategy
        exit_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=self.asset_id,
            target_node=agent_id,
            belief_type="exit_strategy",
            confidence=1.0,
            value={"exit_distribution": {"hold_till_end": 0.4, "sell_early": 0.3, "opportunistic": 0.3}},
            timestamp=self.current_time
        )
        self.edges[exit_edge.edge_id] = exit_edge
        
        # Add self-beliefs for this agent (probabilistic beliefs about own state)
        # These will be dynamically updated based on market observations
        
        # Self-belief about market direction assessment (probabilistic)
        self_direction_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=agent_id,  # Agent has beliefs about themselves
            target_node=agent_id,
            belief_type="self_market_direction",
            confidence=1.0,
            value={"direction_distribution": {"up": 0.33, "down": 0.33, "sideways": 0.34}},
            timestamp=self.current_time
        )
        self.edges[self_direction_edge.edge_id] = self_direction_edge
        
        # Self-belief about optimal entry prices (probabilistic)
        self_price_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=agent_id,
            target_node=agent_id,
            belief_type="self_optimal_entry",
            confidence=1.0,
            value={"price_distribution": {"75": 0.02, "80": 0.02, "85": 0.03, "90": 0.04, "95": 0.05, "100": 0.10, "105": 0.10, "110": 0.10, "115": 0.08, "120": 0.06, "125": 0.05, "130": 0.04, "135": 0.04, "140": 0.03, "145": 0.03, "150": 0.03, "155": 0.03, "160": 0.03, "165": 0.02, "170": 0.02, "175": 0.02, "180": 0.02, "185": 0.02, "190": 0.02, "195": 0.01, "200": 0.01}},
            timestamp=self.current_time
        )
        self.edges[self_price_edge.edge_id] = self_price_edge
        
        # Self-belief about time urgency (probabilistic)
        self_urgency_edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node=agent_id,
            target_node=agent_id,
            belief_type="self_time_urgency",
            confidence=1.0,
            value={"urgency_distribution": {"low": 0.5, "medium": 0.3, "high": 0.2}},
            timestamp=self.current_time
        )
        self.edges[self_urgency_edge.edge_id] = self_urgency_edge
    
    def update_beliefs(self, event: MarketEvent) -> None:
        """
        Update the belief graph based on a market event.
        
        This is the core function that ingests market events and revises
        the belief graph accordingly.
        """
        self.current_time = event.timestamp
        self.event_history.append(event)
        
        # Ensure the agent exists in the graph
        if event.agent_id and event.agent_id not in self.nodes:
            self.add_agent(event.agent_id)
        
        # Update based on event type
        if event.event_type == EventType.BID:
            self._update_beliefs_from_bid(event)
        elif event.event_type == EventType.ASK:
            self._update_beliefs_from_ask(event)
        elif event.event_type == EventType.TRADE:
            self._update_beliefs_from_trade(event)
        elif event.event_type == EventType.CANCEL:
            self._update_beliefs_from_cancel(event)
        
        # Update asset state
        self._update_asset_state()
        
        # Decay old beliefs
        self._decay_old_beliefs()
    
    def _update_beliefs_from_bid(self, event: MarketEvent) -> None:
        """Update beliefs based on a bid event"""
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_bid_price = event.price
        agent_node.last_activity = event.timestamp
        
        # Update discrete beliefs using set elimination logic
        self._update_discrete_beliefs_from_market_event(event.agent_id, event.price, "bid")
    
    def _update_beliefs_from_ask(self, event: MarketEvent) -> None:
        """Update beliefs based on an ask event"""
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_ask_price = event.price
        agent_node.last_activity = event.timestamp
        
        # Update valuation belief
        self._update_valuation_belief(event.agent_id, event.price, "ask")
    
    def _update_beliefs_from_trade(self, event: MarketEvent) -> None:
        """Update beliefs based on a trade event"""
        gv2_logger.debug(f"[BG-DEBUG] _update_beliefs_from_trade called for agent {event.agent_id} at price {event.price}")
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        old_aggr = agent_node.aggressiveness_score
        agent_node.last_trade_price = event.price
        agent_node.total_trades += 1
        agent_node.total_volume += event.quantity or 1
        agent_node.last_activity = event.timestamp
        
        # Update aggressiveness based on trade price compared to previous trades FIRST
        # (so desperation updates can use the aggressiveness score)
        if self.asset_node.last_trade_price and agent_node.total_trades > 1:
            price_ratio = event.price / self.asset_node.last_trade_price
            gv2_logger.debug(f"[BG-TRADE-AGGR] Agent {event.agent_id}: price={event.price}, last_price={self.asset_node.last_trade_price}, ratio={price_ratio:.3f}")
            
            # Buyer perspective (assuming agent_id is buyer in BSE)
            if price_ratio > 1.01:  # Paid >1% more than last trade
                agent_node.aggressiveness_score = min(1.0, agent_node.aggressiveness_score + 0.3)
                gv2_logger.debug(f"[BG-TRADE-AGGR] AGGRESSIVE buyer: paid {price_ratio-1:.1%} more")
            elif price_ratio < 0.99:  # Paid <1% less than last trade  
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.2)
                gv2_logger.debug(f"[BG-TRADE-AGGR] PASSIVE buyer: paid {1-price_ratio:.1%} less")
            else:
                # Near market price - slight shift toward passive
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.05)
                gv2_logger.debug(f"[BG-TRADE-AGGR] NEUTRAL trade")
        else:
            # First trade - random initial aggressiveness
            import random
            agent_node.aggressiveness_score = random.uniform(-0.3, 0.3)
            gv2_logger.debug(f"[BG-TRADE-AGGR] First trade, random initial aggr={agent_node.aggressiveness_score:.3f}")
        
        gv2_logger.debug(f"[BG-TRADE-AGGR] Agent {event.agent_id} aggr: {old_aggr:.3f} -> {agent_node.aggressiveness_score:.3f}")
        
        # Update valuation belief with high confidence (actual trade)
        self._update_valuation_belief(event.agent_id, event.price, "trade", high_confidence=True)
        # Update ALL other GraphVar2 belief types based on trade behavior
        self._update_direction_belief_probabilistic(event.agent_id, event.price, "trade")
        self._update_desperation_belief_probabilistic(event.agent_id, event.price, "trade")
        self._update_cash_belief_probabilistic(event.agent_id, event.price, "trade")
        self._update_exit_strategy_belief_probabilistic(event.agent_id, event.price, "trade")
        
        # Update strategy belief based on trade
        self._update_strategy_belief(event.agent_id, "trade", event.price)
        
        # Also update counterparty if available
        if event.counterparty_id and event.counterparty_id in self.nodes:
            seller_node = self.nodes[event.counterparty_id]
            old_seller_aggr = seller_node.aggressiveness_score
            seller_node.last_trade_price = event.price
            seller_node.total_trades += 1
            seller_node.total_volume += event.quantity or 1
            
            # Seller perspective - opposite of buyer
            if self.asset_node.last_trade_price and seller_node.total_trades > 1:
                price_ratio = event.price / self.asset_node.last_trade_price
                
                if price_ratio < 0.99:  # Sold <1% below last trade
                    seller_node.aggressiveness_score = min(1.0, seller_node.aggressiveness_score + 0.3)
                    gv2_logger.debug(f"[BG-TRADE-AGGR] AGGRESSIVE seller {event.counterparty_id}: sold {1-price_ratio:.1%} below")
                elif price_ratio > 1.01:  # Sold >1% above last trade
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.2)
                    gv2_logger.debug(f"[BG-TRADE-AGGR] PASSIVE seller {event.counterparty_id}: sold {price_ratio-1:.1%} above")
                else:
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.05)
            else:
                # First trade - random initial
                import random
                seller_node.aggressiveness_score = random.uniform(-0.3, 0.3)
                
            gv2_logger.debug(f"[BG-TRADE-AGGR] Seller {event.counterparty_id} aggr: {old_seller_aggr:.3f} -> {seller_node.aggressiveness_score:.3f}")
            
            # Update seller strategy
            self._update_valuation_belief(event.counterparty_id, event.price, "trade", high_confidence=True)
            self._update_strategy_belief(event.counterparty_id, "trade", event.price)
        
        # Update asset state
        self.asset_node.last_trade_price = event.price
        self.asset_node.volume_traded += event.quantity or 1
    
    def _update_beliefs_from_cancel(self, event: MarketEvent) -> None:
        """Update beliefs based on a cancel event"""
        if not event.agent_id:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_activity = event.timestamp
        
        # Cancellation might indicate uncertainty or strategy change
        agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.05)
    
    def _update_valuation_belief(self, agent_id: str, price: float, action_type: str, high_confidence: bool = False) -> None:
        """Update valuation beliefs using Bayesian probability updates"""
        gv2_logger.debug(f"\n[GraphVar2-VALUATION-UPDATE] Agent {agent_id}, Action: {action_type}, Price: {price}")
        
        # Find existing valuation edge
        valuation_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "valuation"):
                valuation_edge = edge
                break
        
        if valuation_edge is None:
            # Create new valuation edge with uniform distribution
            valuation_edge = BeliefEdge(
                edge_id=str(uuid.uuid4()),
                source_node=self.asset_id,
                target_node=agent_id,
                belief_type="valuation",
                confidence=1.0,  # Full confidence in the distribution
                value={"valuation_distribution": {"70": 0.037, "75": 0.037, "80": 0.037, "85": 0.037, "90": 0.037, "95": 0.037, "100": 0.037, "105": 0.037, "110": 0.037, "115": 0.037, "120": 0.037, "125": 0.037, "130": 0.037, "135": 0.037, "140": 0.037, "145": 0.037, "150": 0.037, "155": 0.037, "160": 0.037, "165": 0.037, "170": 0.037, "175": 0.037, "180": 0.037, "185": 0.037, "190": 0.037, "195": 0.037, "200": 0.038}},
                timestamp=self.current_time
            )
            self.edges[valuation_edge.edge_id] = valuation_edge
        
        # Apply Bayesian probability updates based on market action
        if valuation_edge.value and isinstance(valuation_edge.value, dict) and "valuation_distribution" in valuation_edge.value:
            current_dist = valuation_edge.value["valuation_distribution"].copy()
            
            # If the price is not in our distribution, add it with a small initial probability
            price_str = str(int(price))
            if price_str not in current_dist:
                # Add the new price to distribution with small probability
                # Redistribute probabilities
                total_prob = sum(current_dist.values())
                redistrib_factor = 0.95  # Reduce existing probabilities to 95%
                for key in current_dist:
                    current_dist[key] *= redistrib_factor
                current_dist[price_str] = total_prob * 0.05  # Give 5% to the new price
                gv2_logger.info(f"[GraphVar2-NEW-PRICE] Added {price_str} to distribution for {agent_id}")
            
            # Apply Bayesian inference based on observed behavior
            for val_str, prob in current_dist.items():
                val = float(val_str)
                
                # Calculate likelihood of observing this price given the valuation
                if action_type == "bid":
                    # P(bid at price | valuation) - higher for valuations >= bid price
                    if val >= price:
                        likelihood = self.bid_support_likelihood  # Configurable likelihood
                    else:
                        likelihood = self.bid_contradict_likelihood  # Configurable likelihood
                
                elif action_type == "ask":
                    # P(ask at price | valuation) - higher for valuations <= ask price  
                    if val <= price:
                        likelihood = self.ask_support_likelihood  # Configurable likelihood
                    else:
                        likelihood = self.ask_contradict_likelihood  # Configurable likelihood
                
                elif action_type == "trade":
                    # P(trade at price | valuation) - highest for valuations near trade price
                    distance = abs(val - price)
                    if distance <= self.trade_close_distance:
                        likelihood = self.trade_close_likelihood  # Configurable likelihood
                    elif distance <= self.trade_medium_distance:
                        likelihood = self.trade_medium_likelihood  # Configurable likelihood
                    else:
                        likelihood = self.trade_far_likelihood  # Configurable likelihood
                
                # Bayesian update: P(valuation | observation) ∝ P(observation | valuation) × P(valuation)
                current_dist[val_str] = prob * likelihood
            
            # Log before state with more detail
            old_dist = valuation_edge.value.get("valuation_distribution", {}).copy()
            
            # Find the highest probability values before update
            old_max_val = max(old_dist.items(), key=lambda x: x[1]) if old_dist else ("?", 0)
            
            # Normalize probabilities to sum to 1.0
            total_prob = sum(current_dist.values())
            if total_prob > 0:
                for val_str in current_dist:
                    current_dist[val_str] /= total_prob
            
            # Find the highest probability values after update
            new_max_val = max(current_dist.items(), key=lambda x: x[1]) if current_dist else ("?", 0)
            
            # Only log if there was a meaningful change
            if abs(old_max_val[1] - new_max_val[1]) > 0.01 or old_max_val[0] != new_max_val[0]:
                gv2_logger.info(f"[GraphVar2-BELIEF-UPDATE] {agent_id} {action_type.upper()} at {price}:")
                gv2_logger.info(f"  BEFORE peak: value={old_max_val[0]} prob={old_max_val[1]:.3f}")
                gv2_logger.info(f"  AFTER peak:  value={new_max_val[0]} prob={new_max_val[1]:.3f}")
                gv2_logger.info(f"  Full distribution shift:")
                for val_str in sorted(current_dist.keys(), key=lambda x: float(x)):
                    old_p = old_dist.get(val_str, 0)
                    new_p = current_dist[val_str]
                    if abs(old_p - new_p) > 0.01:  # Only show values that changed significantly
                        gv2_logger.info(f"    {val_str}: {old_p:.3f} -> {new_p:.3f} ({'+' if new_p > old_p else ''}{new_p-old_p:.3f})")
            
            # Update the edge with new probability distribution
            valuation_edge.value = {"valuation_distribution": current_dist}
            valuation_edge.timestamp = self.current_time
            valuation_edge.evidence_count += 1
            
            # Sync agent node with expected value from distribution
            agent_node = self.nodes[agent_id]
            # Calculate expected value (weighted average)
            expected_val = sum(float(val_str) * prob for val_str, prob in current_dist.items())
            # Calculate confidence as inverse of variance (more concentrated = higher confidence)
            variance = sum(prob * (float(val_str) - expected_val)**2 for val_str, prob in current_dist.items())
            confidence = 1.0 / (1.0 + variance / 100.0)  # Normalize variance
            
            agent_node.inferred_valuation = expected_val
            agent_node.valuation_confidence = confidence

    def _update_direction_belief_probabilistic(self, agent_id: str, price: float, action_type: str) -> None:
        """Update probabilistic market direction beliefs based on trade behavior"""
        direction_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "market_direction"):
                direction_edge = edge
                break
        
        if direction_edge and "direction_distribution" in direction_edge.value:
            current_dist = direction_edge.value["direction_distribution"].copy()
            old_dist = current_dist.copy()
            
            # Update based on price trend
            if self.asset_node.last_trade_price:
                if price > self.asset_node.last_trade_price * 1.02:  # Price going up
                    current_dist["up"] *= 1.3
                    current_dist["down"] *= 0.7
                elif price < self.asset_node.last_trade_price * 0.98:  # Price going down  
                    current_dist["down"] *= 1.3
                    current_dist["up"] *= 0.7
                else:  # Sideways movement
                    current_dist["sideways"] *= 1.2
                    
                # Normalize
                total = sum(current_dist.values())
                if total > 0:
                    for key in current_dist:
                        current_dist[key] /= total
                        
                # Log significant changes
                if max(abs(current_dist[k] - old_dist[k]) for k in current_dist) > 0.05:
                    gv2_logger.info(f"[GraphVar2-DIRECTION-UPDATE] {agent_id} {action_type.upper()} at {price}:")
                    for direction in current_dist:
                        old_p = old_dist[direction]
                        new_p = current_dist[direction]
                        if abs(old_p - new_p) > 0.02:
                            gv2_logger.info(f"  {direction}: {old_p:.3f} -> {new_p:.3f} ({'+' if new_p > old_p else ''}{new_p-old_p:.3f})")
                
                direction_edge.value["direction_distribution"] = current_dist
                direction_edge.timestamp = self.current_time
                direction_edge.evidence_count += 1
    
    def _update_desperation_belief_probabilistic(self, agent_id: str, price: float, action_type: str) -> None:
        """Update probabilistic desperation beliefs based on trading behavior"""
        desperation_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "desperation_level"):
                desperation_edge = edge
                break
        
        if desperation_edge and "desperation_distribution" in desperation_edge.value:
            current_dist = desperation_edge.value["desperation_distribution"].copy()
            old_dist = current_dist.copy()
            
            # Analyze trading behavior for desperation signals
            agent_node = self.nodes[agent_id]
            gv2_logger.debug(f"[GraphVar2-DESPERATION-AGGR] {agent_id} aggressiveness_score={agent_node.aggressiveness_score}")
            
            # High aggressiveness suggests desperation
            if agent_node.aggressiveness_score > 0.1:  # Lowered threshold
                current_dist["desperate"] *= 1.5
                current_dist["calm"] *= 0.6
                gv2_logger.debug(f"[GraphVar2-DESPERATION-LOGIC] {agent_id} aggressive, boosting desperate")
            elif agent_node.aggressiveness_score < -0.1:  # Lowered threshold
                current_dist["calm"] *= 1.4
                current_dist["desperate"] *= 0.7
                gv2_logger.debug(f"[GraphVar2-DESPERATION-LOGIC] {agent_id} passive, boosting calm")
            else:
                current_dist["moderate"] *= 1.2
                gv2_logger.debug(f"[GraphVar2-DESPERATION-LOGIC] {agent_id} moderate, boosting moderate")
                
            # Normalize
            total = sum(current_dist.values())
            if total > 0:
                for key in current_dist:
                    current_dist[key] /= total
                    
            # Log significant changes
            if max(abs(current_dist[k] - old_dist[k]) for k in current_dist) > 0.05:
                gv2_logger.info(f"[GraphVar2-DESPERATION-UPDATE] {agent_id} {action_type.upper()} at {price}:")
                for desp in current_dist:
                    old_p = old_dist[desp]
                    new_p = current_dist[desp]
                    if abs(old_p - new_p) > 0.02:
                        gv2_logger.info(f"  {desp}: {old_p:.3f} -> {new_p:.3f} ({'+' if new_p > old_p else ''}{new_p-old_p:.3f})")
            
            desperation_edge.value["desperation_distribution"] = current_dist
            desperation_edge.timestamp = self.current_time
            desperation_edge.evidence_count += 1
    
    def _update_cash_belief_probabilistic(self, agent_id: str, price: float, action_type: str) -> None:
        """Update probabilistic cash beliefs based on trading volume"""
        cash_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "available_cash"):
                cash_edge = edge
                break
        
        if cash_edge and "cash_distribution" in cash_edge.value:
            current_dist = cash_edge.value["cash_distribution"].copy()
            old_dist = current_dist.copy()
            
            # Analyze volume for cash inference
            agent_node = self.nodes[agent_id]
            
            # High volume suggests more cash
            if agent_node.total_volume >= 3:
                current_dist["high"] *= 1.4
                current_dist["low"] *= 0.6
            elif agent_node.total_volume >= 2:
                current_dist["medium"] *= 1.3
            else:
                current_dist["low"] *= 1.2
                current_dist["high"] *= 0.8
                
            # Normalize
            total = sum(current_dist.values())
            if total > 0:
                for key in current_dist:
                    current_dist[key] /= total
                    
            # Log significant changes
            if max(abs(current_dist[k] - old_dist[k]) for k in current_dist) > 0.05:
                gv2_logger.info(f"[GraphVar2-CASH-UPDATE] {agent_id} {action_type.upper()} at {price} (volume={agent_node.total_volume}):")
                for cash in current_dist:
                    old_p = old_dist[cash]
                    new_p = current_dist[cash]
                    if abs(old_p - new_p) > 0.02:
                        gv2_logger.info(f"  {cash}: {old_p:.3f} -> {new_p:.3f} ({'+' if new_p > old_p else ''}{new_p-old_p:.3f})")
            
            cash_edge.value["cash_distribution"] = current_dist
            cash_edge.timestamp = self.current_time
            cash_edge.evidence_count += 1
    
    def _update_exit_strategy_belief_probabilistic(self, agent_id: str, price: float, action_type: str) -> None:
        """Update probabilistic exit strategy beliefs based on trading patterns"""
        exit_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "exit_strategy"):
                exit_edge = edge
                break
        
        if exit_edge and "exit_distribution" in exit_edge.value:
            current_dist = exit_edge.value["exit_distribution"].copy()
            old_dist = current_dist.copy()
            
            # Analyze trading frequency for exit strategy
            agent_node = self.nodes[agent_id]
            
            # Frequent trading suggests opportunistic strategy
            if agent_node.total_trades >= 3:
                current_dist["opportunistic"] *= 1.5
                current_dist["hold_till_end"] *= 0.6
            elif agent_node.total_trades >= 2:
                current_dist["sell_early"] *= 1.3
            else:
                current_dist["hold_till_end"] *= 1.2
                current_dist["opportunistic"] *= 0.8
                
            # Normalize
            total = sum(current_dist.values())
            if total > 0:
                for key in current_dist:
                    current_dist[key] /= total
                    
            # Log significant changes
            if max(abs(current_dist[k] - old_dist[k]) for k in current_dist) > 0.05:
                gv2_logger.info(f"[GraphVar2-EXIT-UPDATE] {agent_id} {action_type.upper()} at {price} (trades={agent_node.total_trades}):")
                for exit_strat in current_dist:
                    old_p = old_dist[exit_strat]
                    new_p = current_dist[exit_strat]
                    if abs(old_p - new_p) > 0.02:
                        gv2_logger.info(f"  {exit_strat}: {old_p:.3f} -> {new_p:.3f} ({'+' if new_p > old_p else ''}{new_p-old_p:.3f})")
            
            exit_edge.value["exit_distribution"] = current_dist
            exit_edge.timestamp = self.current_time
            exit_edge.evidence_count += 1

    def _update_strategy_belief(self, agent_id: str, action_type: str, price: float) -> None:
        """Update the belief about an agent's strategy"""
        gv2_logger.debug(f"[BG-STRAT-START] Updating strategy for {agent_id}, action={action_type}, price={price}")
        
        # Find existing strategy edge
        strategy_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "strategy"):
                strategy_edge = edge
                break
        
        if strategy_edge is None:
            gv2_logger.debug(f"[BG-STRAT-ERROR] No strategy edge found for {agent_id}")
            return
        
        # Simple strategy classification based on behavior patterns
        agent_node = self.nodes[agent_id]
        
        gv2_logger.debug(f"[BG-STRAT-INFO] Agent {agent_id}: trades={agent_node.total_trades}, aggr_score={agent_node.aggressiveness_score:.3f}, current_strategy={agent_node.strategy_type}")
        
        # Always classify based on aggressiveness, even if no trades yet
        old_strategy = agent_node.strategy_type
        
        # Use more sensitive thresholds
        if agent_node.aggressiveness_score > 0.15:
            strategy = "aggressive"
        elif agent_node.aggressiveness_score < -0.15:
            strategy = "passive"
        else:
            strategy = "neutral"
        
        gv2_logger.debug(f"[BG-STRAT-CLASSIFY] Agent {agent_id}: aggr={agent_node.aggressiveness_score:.3f} -> strategy={strategy} (was {old_strategy})")
        
        strategy_edge.value = strategy
        strategy_edge.confidence = min(self.max_confidence, strategy_edge.confidence + 0.1)
        strategy_edge.timestamp = self.current_time
        strategy_edge.evidence_count += 1
        
        # Sync the agent node's strategy_type field
        agent_node.strategy_type = strategy
        
        gv2_logger.debug(f"[BG-STRAT-END] Agent {agent_id} strategy updated to {strategy} (confidence={strategy_edge.confidence:.2f})")
    
    def _update_asset_state(self) -> None:
        """Update the asset node state based on current market conditions"""
        # This would typically be called with actual market data
        # For now, we'll update based on the belief graph state
        
        # Calculate spread if we have both bid and ask
        if (self.asset_node.current_best_bid is not None and 
            self.asset_node.current_best_ask is not None):
            self.asset_node.spread_width = self.asset_node.current_best_ask - self.asset_node.current_best_bid
    
    def _decay_old_beliefs(self) -> None:
        """Decay confidence in old beliefs"""
        current_time = self.current_time
        for edge in self.edges.values():
            time_diff = current_time - edge.timestamp
            if time_diff > 100:  # Decay beliefs older than 100 time units
                decay_factor = self.valuation_decay_rate ** (time_diff / 100)
                edge.confidence *= decay_factor
    
    
    def _update_self_beliefs(self, agent_id: str, market_state: Dict[str, Any]) -> None:
        """Update agent's probabilistic beliefs about their own market assessment and strategy"""
        # Find self-belief edges for this agent
        self_direction_edge = None
        self_price_edge = None
        self_urgency_edge = None
        
        for edge in self.edges.values():
            if (edge.source_node == agent_id and edge.target_node == agent_id):
                if edge.belief_type == "self_market_direction":
                    self_direction_edge = edge
                elif edge.belief_type == "self_optimal_entry":
                    self_price_edge = edge
                elif edge.belief_type == "self_time_urgency":
                    self_urgency_edge = edge
        
        # Update market direction belief using Bayesian updates
        if self_direction_edge and self_direction_edge.value:
            direction_dist = self_direction_edge.value.get("direction_distribution", {"up": 0.33, "down": 0.33, "sideways": 0.34}).copy()
            
            # Analyze recent price trend and update probabilities
            if 'best_bid' in market_state and 'best_ask' in market_state:
                bid, ask = market_state['best_bid'], market_state['best_ask']
                if bid and ask:
                    spread = ask - bid
                    # Wide spread indicates uncertainty -> increase sideways probability
                    # Narrow spread indicates direction -> increase up/down probabilities
                    if spread > 15:  # Wide spread
                        direction_dist["sideways"] *= 1.2  # Boost sideways
                        direction_dist["up"] *= 0.9      # Reduce up
                        direction_dist["down"] *= 0.9    # Reduce down
                    elif spread < 5:  # Narrow spread
                        direction_dist["sideways"] *= 0.8  # Reduce sideways
                        direction_dist["up"] *= 1.1      # Boost up
                        direction_dist["down"] *= 1.1    # Boost down
                    
                    # Normalize probabilities
                    total = sum(direction_dist.values())
                    if total > 0:
                        for key in direction_dist:
                            direction_dist[key] /= total
                    
            self_direction_edge.value = {"direction_distribution": direction_dist}
            self_direction_edge.timestamp = self.current_time
        
        # Update price belief using Bayesian updates
        if self_price_edge and self_price_edge.value:
            price_dist = self_price_edge.value.get("price_distribution", {"85": 0.1, "90": 0.15, "95": 0.2, "100": 0.3, "105": 0.15, "110": 0.1}).copy()
            
            # Focus probability on prices near current market
            if 'best_bid' in market_state and 'best_ask' in market_state:
                bid, ask = market_state['best_bid'], market_state['best_ask']
                if bid and ask:
                    market_mid = (bid + ask) / 2
                    
                    # Bayesian update: boost probabilities for prices near market mid
                    for price_str, prob in price_dist.items():
                        price_val = float(price_str)
                        distance = abs(price_val - market_mid)
                        
                        # Likelihood function: closer prices are more likely
                        if distance <= 5:
                            likelihood = 1.5  # High likelihood for close prices
                        elif distance <= 10:
                            likelihood = 1.0  # Medium likelihood
                        else:
                            likelihood = 0.7  # Lower likelihood for distant prices
                        
                        price_dist[price_str] = prob * likelihood
                    
                    # Normalize probabilities
                    total = sum(price_dist.values())
                    if total > 0:
                        for key in price_dist:
                            price_dist[key] /= total
                    
            self_price_edge.value = {"price_distribution": price_dist}
            self_price_edge.timestamp = self.current_time
        
        # Update urgency belief using Bayesian updates based on time
        if self_urgency_edge and self_urgency_edge.value:
            urgency_dist = self_urgency_edge.value.get("urgency_distribution", {"low": 0.5, "medium": 0.3, "high": 0.2}).copy()
            
            # Time-based Bayesian update
            time_remaining = market_state.get('time_remaining', 200)
            
            # Likelihood of each urgency level given time remaining
            if time_remaining < 30:  # Very urgent
                urgency_dist["low"] *= 0.1    # Very unlikely to be low urgency
                urgency_dist["medium"] *= 0.5  # Less likely medium
                urgency_dist["high"] *= 2.0    # Much more likely high urgency
            elif time_remaining < 100:  # Moderate urgency
                urgency_dist["low"] *= 0.5    # Less likely low
                urgency_dist["medium"] *= 1.5  # More likely medium
                urgency_dist["high"] *= 1.2    # Somewhat more likely high
            
            # Normalize probabilities
            total = sum(urgency_dist.values())
            if total > 0:
                for key in urgency_dist:
                    urgency_dist[key] /= total
            
            self_urgency_edge.value = {"urgency_distribution": urgency_dist}
            self_urgency_edge.timestamp = self.current_time

    def _get_dynamic_self_beliefs(self, agent_id: str, current_market_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get agent's probabilistic self-beliefs from stored edges"""
        # Find self-belief edges for this agent
        self_direction_edge = None
        self_price_edge = None
        self_urgency_edge = None
        
        for edge in self.edges.values():
            if (edge.source_node == agent_id and edge.target_node == agent_id):
                if edge.belief_type == "self_market_direction":
                    self_direction_edge = edge
                elif edge.belief_type == "self_optimal_entry":
                    self_price_edge = edge
                elif edge.belief_type == "self_time_urgency":
                    self_urgency_edge = edge
        
        # Generate actual situation descriptions based on market state
        actual_signals = "mixed_price_movements"
        if 'best_bid' in current_market_state and 'best_ask' in current_market_state:
            bid, ask = current_market_state.get('best_bid'), current_market_state.get('best_ask')
            if bid and ask:
                spread = ask - bid
                if spread > 15:
                    actual_signals = "wide_spread_uncertainty"
                elif spread < 5:
                    actual_signals = "tight_spread_consensus"
                else:
                    actual_signals = "moderate_spread_activity"
        
        actual_situation = "volatile_spreads"
        if 'best_bid' in current_market_state and 'best_ask' in current_market_state:
            bid, ask = current_market_state.get('best_bid'), current_market_state.get('best_ask')
            if bid and ask:
                market_mid = (bid + ask) / 2
                if market_mid < 95:
                    actual_situation = "low_market_prices"
                elif market_mid > 105:
                    actual_situation = "high_market_prices"
                else:
                    actual_situation = "balanced_market_prices"
        
        actual_urgency_situation = "time_pressure_building"
        time_remaining = current_market_state.get('time_remaining', 200)
        if time_remaining < 50:
            actual_urgency_situation = "urgent_time_pressure"
        elif time_remaining < 100:
            actual_urgency_situation = "moderate_time_pressure"
        else:
            actual_urgency_situation = "comfortable_time_remaining"
        
        return {
            "My_Market_Direction": {
                "actual_signals_I_see": actual_signals,
                "my_self_belief": {
                    "direction_distribution": self_direction_edge.value.get("direction_distribution", {"up": 0.33, "down": 0.33, "sideways": 0.34}) if self_direction_edge and self_direction_edge.value else {"up": 0.33, "down": 0.33, "sideways": 0.34}
                }
            },
            "My_Optimal_Entry_Price": {
                "actual_situation_I_face": actual_situation,
                "my_self_belief": {
                    "price_distribution": self_price_edge.value.get("price_distribution", {"85": 0.1, "90": 0.15, "95": 0.2, "100": 0.3, "105": 0.15, "110": 0.1}) if self_price_edge and self_price_edge.value else {"85": 0.1, "90": 0.15, "95": 0.2, "100": 0.3, "105": 0.15, "110": 0.1}
                }
            },
            "My_Time_Urgency": {
                "actual_situation_I_face": actual_urgency_situation,
                "my_self_belief": {
                    "urgency_distribution": self_urgency_edge.value.get("urgency_distribution", {"low": 0.5, "medium": 0.3, "high": 0.2}) if self_urgency_edge and self_urgency_edge.value else {"low": 0.5, "medium": 0.3, "high": 0.2}
                }
            }
        }
    
    def query_action(self, agent_id: str, current_market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the belief graph for decision-making.
        
        This function serializes the belief graph and returns it in a format
        suitable for LLM processing.
        """
        # Update asset state with current market data
        if 'best_bid' in current_market_state:
            self.asset_node.current_best_bid = current_market_state['best_bid']
        if 'best_ask' in current_market_state:
            self.asset_node.current_best_ask = current_market_state['best_ask']
        if 'last_trade' in current_market_state:
            self.asset_node.last_trade_price = current_market_state['last_trade']
        
        # Generate competitor beliefs using actual stored probability distributions
        competitor_beliefs = {}
        for node_id, node in self.nodes.items():
            if isinstance(node, AgentNode) and node_id != agent_id:
                # Observe behavior with rich context
                observed_behavior = "passive_observation"
                if node.last_bid_price:
                    # Generate contextual bid behavior
                    current_ask = self.asset_node.current_best_ask
                    if current_ask and node.last_bid_price >= current_ask * 0.95:
                        observed_behavior = f"aggressive_bidding_at_{node.last_bid_price}"
                    elif node.aggressiveness_score > 0.1:
                        observed_behavior = f"active_bidding_at_{node.last_bid_price}"
                    else:
                        observed_behavior = f"cautious_bidding_at_{node.last_bid_price}"
                elif node.last_ask_price:
                    # Generate contextual ask behavior  
                    current_bid = self.asset_node.current_best_bid
                    if current_bid and node.last_ask_price <= current_bid * 1.05:
                        observed_behavior = f"aggressive_asking_at_{node.last_ask_price}"
                    elif node.aggressiveness_score > 0.1:
                        observed_behavior = f"active_asking_at_{node.last_ask_price}"
                    else:
                        observed_behavior = f"cautious_asking_at_{node.last_ask_price}"
                elif node.last_trade_price:
                    # Trading behavior context
                    if node.total_trades >= 3:
                        observed_behavior = f"frequent_trading_at_{node.last_trade_price}"
                    elif node.aggressiveness_score > 0.2:
                        observed_behavior = f"aggressive_trading_at_{node.last_trade_price}"
                    else:
                        observed_behavior = f"careful_trading_at_{node.last_trade_price}"
                
                competitor_beliefs[f"{node_id}_Beliefs"] = {}
                
                # Get all belief types for this competitor from stored probability distributions
                for edge in self.edges.values():
                    if edge.target_node == node_id and edge.value and isinstance(edge.value, dict):
                        
                        if edge.belief_type == "valuation" and "valuation_distribution" in edge.value:
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Valuation"] = {
                                "actual_behavior_I_observe": observed_behavior,
                                f"{node_id.lower()}_self_belief": {
                                    "valuation_distribution": edge.value["valuation_distribution"]
                                }
                            }
                        
                        elif edge.belief_type == "market_direction" and "direction_distribution" in edge.value:
                            # Rich timing behavior context
                            if node.aggressiveness_score <= -0.1:
                                timing_behavior = "very_cautious_timing"
                            elif node.aggressiveness_score <= 0.1:
                                timing_behavior = "cautious_timing"
                            elif node.aggressiveness_score <= 0.3:
                                timing_behavior = "moderate_timing"
                            elif node.aggressiveness_score <= 0.6:
                                timing_behavior = "urgent_timing"
                            else:
                                timing_behavior = "extremely_urgent_timing"
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Market_Direction"] = {
                                "actual_behavior_I_observe": timing_behavior,
                                f"{node_id.lower()}_self_belief": {
                                    "direction_distribution": edge.value["direction_distribution"]
                                }
                            }
                        
                        elif edge.belief_type == "desperation_level" and "desperation_distribution" in edge.value:
                            # Rich desperation behavior context
                            if node.aggressiveness_score <= -0.2:
                                desperation_behavior = "extremely_patient_small_bids"
                            elif node.aggressiveness_score <= 0.0:
                                desperation_behavior = "patient_small_bids"
                            elif node.aggressiveness_score <= 0.2:
                                desperation_behavior = "moderate_sized_orders"
                            elif node.aggressiveness_score <= 0.4:
                                desperation_behavior = "moderate_pressure_bids"
                            elif node.aggressiveness_score <= 0.7:
                                desperation_behavior = "urgent_larger_orders"
                            else:
                                desperation_behavior = "large_urgent_orders"
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Desperation_Level"] = {
                                "actual_behavior_I_observe": desperation_behavior,
                                f"{node_id.lower()}_self_belief": {
                                    "desperation_distribution": edge.value["desperation_distribution"]
                                }
                            }
                        
                        elif edge.belief_type == "available_cash" and "cash_distribution" in edge.value:
                            # Rich cash behavior context
                            if node.total_volume <= 1:
                                cash_behavior = "very_small_consistent_volumes"
                            elif node.total_volume <= 3:
                                cash_behavior = "consistent_small_volumes"
                            elif node.total_volume <= 6:
                                cash_behavior = "moderate_position_sizes"
                            elif node.total_volume <= 10:
                                cash_behavior = "larger_position_sizes"
                            else:
                                cash_behavior = "very_large_position_sizes"
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Available_Cash"] = {
                                "actual_behavior_I_observe": cash_behavior,
                                f"{node_id.lower()}_self_belief": {
                                    "cash_distribution": edge.value["cash_distribution"]
                                }
                            }
                        
                        elif edge.belief_type == "exit_strategy" and "exit_distribution" in edge.value:
                            # Rich exit strategy behavior context
                            if node.total_trades == 0:
                                exit_behavior = "no_trading_activity"
                            elif node.total_trades == 1:
                                exit_behavior = "minimal_trading"
                            elif node.total_trades <= 2:
                                exit_behavior = "holding_positions"
                            elif node.total_trades <= 4:
                                exit_behavior = "moderate_trading"
                            elif node.total_trades <= 7:
                                exit_behavior = "active_trading"
                            else:
                                exit_behavior = "very_active_trading"
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Exit_Strategy"] = {
                                "actual_behavior_I_observe": exit_behavior,
                                f"{node_id.lower()}_self_belief": {
                                    "exit_distribution": edge.value["exit_distribution"]
                                }
                            }
                    
        # GraphVar2 returns pure probabilistic structures
        return {
            "MarketState": {
                "current_bid": current_market_state.get('best_bid'),
                "current_ask": current_market_state.get('best_ask'), 
                "time_remaining": current_market_state.get('time_remaining', 0)
            },
            "My_Trading_Beliefs": self._get_dynamic_self_beliefs(agent_id, current_market_state),
            "Competitor_Trading_Beliefs": competitor_beliefs
        }
    def _generate_strategic_insights(self, agent_id: str) -> Dict[str, Any]:
        """Generate strategic insights for the querying agent"""
        insights = {
            'competitors': [],
            'market_opportunities': [],
            'risk_factors': []
        }
        
        # Analyze competitors
        for node_id, node in self.nodes.items():
            if isinstance(node, AgentNode) and node_id != agent_id:
                competitor_info = {
                    'agent_id': node_id,
                    'strategy': node.strategy_type or "unknown",
                    'aggressiveness': node.aggressiveness_score,
                    'valuation_estimate': node.inferred_valuation,
                    'confidence': node.valuation_confidence,
                    'recent_activity': node.last_activity
                }
                insights['competitors'].append(competitor_info)
        
        # Identify market opportunities
        if (self.asset_node.current_best_bid is not None and 
            self.asset_node.current_best_ask is not None):
            spread = self.asset_node.current_best_ask - self.asset_node.current_best_bid
            if spread > 5:  # Arbitrage opportunity
                insights['market_opportunities'].append({
                    'type': 'arbitrage',
                    'spread': spread,
                    'description': f"Large spread of {spread} points"
                })
        
        # Identify risk factors
        if self.asset_node.price_volatility > 0.1:
            insights['risk_factors'].append({
                'type': 'high_volatility',
                'value': self.asset_node.price_volatility,
                'description': "High price volatility detected"
            })
        
        return insights
    
    def to_json(self) -> str:
        """Serialize the belief graph to JSON"""
        graph_data = {
            'graph_id': self.graph_id,
            'asset_id': self.asset_id,
            'current_time': self.current_time,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            'event_history': [event.to_dict() for event in self.event_history[-50:]]  # Last 50 events
        }
        return json.dumps(graph_data, indent=2)
    
    def from_json(self, json_str: str) -> None:
        """Deserialize the belief graph from JSON"""
        data = json.loads(json_str)
        self.graph_id = data['graph_id']
        self.asset_id = data['asset_id']
        self.current_time = data['current_time']
        
        # Reconstruct nodes
        self.nodes.clear()
        for node_id, node_data in data['nodes'].items():
            if node_data['node_type'] == NodeType.AGENT.value:
                self.nodes[node_id] = AgentNode(**node_data)
            elif node_data['node_type'] == NodeType.ASSET.value:
                self.nodes[node_id] = AssetNode(**node_data)
        
        # Reconstruct edges
        self.edges.clear()
        for edge_id, edge_data in data['edges'].items():
            self.edges[edge_id] = BeliefEdge(**edge_data)
        
        # Reconstruct event history
        self.event_history = [MarketEvent(**event_data) for event_data in data['event_history']]
    
    def get_agent_beliefs(self, agent_id: str) -> Dict[str, Any]:
        """Get all beliefs about a specific agent"""
        beliefs = {}
        
        # First get the raw belief edges
        for edge in self.edges.values():
            if edge.target_node == agent_id:
                if edge.belief_type not in beliefs:
                    beliefs[edge.belief_type] = []
                beliefs[edge.belief_type].append(edge.to_dict())
        
        # Extract valuation estimate if available
        valuation_estimate = None
        if agent_id in self.nodes:
            agent_node = self.nodes[agent_id]
            if hasattr(agent_node, 'inferred_valuation'):
                valuation_estimate = agent_node.inferred_valuation
            
            # Also try to get from valuation edge
            for edge in self.edges.values():
                if (edge.target_node == agent_id and 
                    edge.belief_type == "valuation" and 
                    edge.value and isinstance(edge.value, dict)):
                    possible_vals = edge.value.get("possible_valuations", [])
                    if possible_vals:
                        # Use median of possible valuations as estimate
                        valuation_estimate = sorted(possible_vals)[len(possible_vals)//2]
                        break
        
        # Extract strategy type if available
        strategy_type = None
        if agent_id in self.nodes:
            agent_node = self.nodes[agent_id]
            if hasattr(agent_node, 'strategy_type'):
                strategy_type = agent_node.strategy_type
        
        # Add simplified summary
        beliefs['valuation_estimate'] = valuation_estimate
        beliefs['strategy_type'] = strategy_type
        
        return beliefs
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get a summary of the current market state"""
        return {
            'asset_state': self.asset_node.to_dict(),
            'active_agents': len([n for n in self.nodes.values() if isinstance(n, AgentNode)]),
            'total_beliefs': len(self.edges),
            'recent_events': len(self.event_history),
            'current_time': self.current_time
        }

# Example usage and testing functions
def create_sample_belief_graph() -> BeliefGraph:
    """Create a sample belief graph for testing"""
    graph = BeliefGraph(asset_id="BTC_USD")
    
    # Add some agents
    agents = ["Alice", "Bob", "Charlie", "Diana"]
    for agent in agents:
        graph.add_agent(agent)
    
    # Simulate some market events
    events = [
        MarketEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.BID,
            timestamp=1.0,
            agent_id="Alice",
            price=100.0,
            quantity=1
        ),
        MarketEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ASK,
            timestamp=2.0,
            agent_id="Bob",
            price=105.0,
            quantity=1
        ),
        MarketEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.TRADE,
            timestamp=3.0,
            agent_id="Alice",
            price=102.0,
            quantity=1,
            counterparty_id="Bob"
        )
    ]
    
    for event in events:
        graph.update_beliefs(event)
    
    return graph





class PerfectBeliefGraph:
    """
    PERFECT BELIEF GRAPH with GROUND TRUTH ACCESS to all trader internals!
    
    This version cheats by directly accessing trader objects to get:
    - EXACT strategy types (ZIP, PT1, SHVR, etc.)
    - REAL internal parameters (margins, inventory, purchase prices)
    - PERFECT valuations (actual limit prices from orders)
    - DETERMINISTIC behavior predictions (next actions for predictable traders)
    
    This is the ULTIMATE TEST of belief→action translation!
    """
    
    def __init__(self, asset_id: str = "DEFAULT_ASSET", traders_dict: Dict[str, Any] = None):
        self.graph_id = str(uuid.uuid4())
        self.asset_id = asset_id
        self.nodes: Dict[str, AgentNode | AssetNode] = {}
        self.edges: Dict[str, BeliefEdge] = {}
        self.event_history: List[MarketEvent] = []
        self.current_time = 0.0
        
        # PERFECT INFORMATION ACCESS - Direct trader object references!
        self.traders_dict = traders_dict or {}  # Access to ALL trader objects
        self.perfect_info_enabled = True
        
        # Initialize the asset node
        self.asset_node = AssetNode(asset_id=asset_id)
        self.nodes[asset_id] = self.asset_node
        
        # Perfect belief parameters (higher confidence, no decay)
        self.valuation_decay_rate = 1.0   # No decay for perfect info
        self.confidence_boost = 0.0       # No gradual learning needed
        self.max_confidence = 1.0         # Perfect confidence possible
        
    def add_agent(self, agent_id: str) -> None:
        """Add agent with PERFECT INFORMATION from direct trader access"""
        if agent_id not in self.nodes:
            agent_node = AgentNode(agent_id=agent_id)
            self.nodes[agent_id] = agent_node
            
            # Add PERFECT beliefs by directly accessing trader object
            self._add_perfect_beliefs(agent_id)
    
    def _add_perfect_beliefs(self, agent_id: str) -> None:
        """
        Add ONLY the original 2 belief types with PERFECT VALUES instead of inferred ones.
        
        ORIGINAL STRUCTURE MAINTAINED - only strategy and valuation beliefs exist.
        We just cheat by getting perfect values instead of inferring them from behavior.
        """
        trader = self.traders_dict.get(agent_id)
        if not trader:
            raise ValueError(f"No trader found for agent_id: {agent_id}")
            
        # PERFECT STRATEGY - provide extremely detailed behavioral description from BSE.py documentation
        # Original only had: "unknown", "aggressive", "passive", "neutral"  
        # We cheat by providing complete business logic descriptions that LLM can act on
        if trader.ttype == "ZIC":
            strategy_desc = "zero_intelligence_constrained_random_trader_that_picks_random_prices_between_system_min_max_but_respects_limit_price_constraints_never_trading_at_loss_purely_random_within_profitable_bounds_no_learning_or_adaptation_just_uniform_random_quote_generation"
        elif trader.ttype == "SHVR":
            strategy_desc = "aggressive_price_improvement_shaver_that_always_tries_to_beat_current_best_price_by_exactly_one_penny_if_bids_exist_quotes_best_bid_plus_1_if_asks_exist_quotes_best_ask_minus_1_but_never_exceeds_own_limit_price_creates_stub_quotes_at_system_extremes_when_no_competition_exists_maximally_competitive_minimal_profit_margin_strategy"
        elif trader.ttype == "SNPR":
            strategy_desc = "time_sensitive_sniping_trader_that_lurks_passively_early_in_session_but_becomes_increasingly_aggressive_as_countdown_timer_approaches_zero_starts_conservative_then_ramps_up_urgency_willing_to_accept_worse_prices_as_time_pressure_mounts_designed_to_capture_last_minute_trading_opportunities"
        elif trader.ttype == "ZIP":
            strategy_desc = "zero_intelligence_plus_adaptive_learning_trader_with_dynamic_profit_margins_that_learns_from_market_feedback_adjusts_bid_sell_margins_based_on_recent_success_failure_uses_momentum_and_learning_rates_to_evolve_strategy_over_time_starts_with_random_margins_then_optimizes_through_reinforcement_learning_from_accepted_rejected_quotes"
        elif trader.ttype == "PT1":
            strategy_desc = "proprietary_buy_and_hold_value_trader_that_waits_5_minutes_for_prices_to_settle_then_buys_when_best_ask_is_below_recent_transaction_price_average_holds_inventory_until_can_sell_at_purchase_price_plus_fixed_profit_margin_long_only_strategy_with_patience_driven_value_investing_approach_only_trades_when_confident_of_profit"
        elif trader.ttype == "PT2":
            strategy_desc = "enhanced_proprietary_buy_and_hold_trader_similar_to_PT1_but_with_refined_parameters_for_bid_percentage_and_profit_margins_also_waits_for_market_settlement_buys_undervalued_assets_based_on_recent_price_history_holds_for_profitable_exit_but_with_different_risk_tolerance_and_patience_thresholds_than_PT1"
        elif trader.ttype == "PRZI":
            strategy_desc = "parameterized_response_zero_intelligence_trader_that_uses_clifford_response_function_to_determine_quote_prices_based_on_mathematical_model_of_market_conditions_adjusts_aggressiveness_based_on_order_book_state_and_internal_parameters_more_sophisticated_than_basic_ZIC_but_still_follows_deterministic_rules_rather_than_learning"
        elif trader.ttype == "GVWY":
            strategy_desc = "giveaway_trader_that_is_even_more_passive_than_zero_intelligence_always_quotes_at_limit_price_essentially_giving_away_profits_to_counterparties_but_never_trading_at_a_loss_minimal_competitiveness_maximum_generosity_while_still_respecting_profitability_constraints_designed_as_market_maker_of_last_resort"
        elif trader.ttype == "BG":
            strategy_desc = "belief_graph_enabled_large_language_model_trader_that_maintains_dynamic_beliefs_about_other_market_participants_strategies_valuations_and_behaviors_uses_AI_reasoning_to_make_trading_decisions_based_on_inferred_competitor_models_and_market_state_representation_this_is_us_the_current_trader_type"
        elif trader.ttype == "LLMHM":
            strategy_desc = "hypothetical_minds_large_language_model_trader_that_generates_hypothetical_scenarios_about_market_participant_behaviors_and_tests_strategies_against_simulated_outcomes_uses_counterfactual_reasoning_and_what_if_analysis_to_inform_trading_decisions_advanced_AI_with_predictive_modeling_capabilities"
        else:
            strategy_desc = "unknown_strategy_type_with_unspecified_business_logic_and_trading_behavior_patterns_may_have_custom_implementation_not_covered_by_standard_BSE_trader_types_proceed_with_caution_and_observe_actual_trading_patterns_to_infer_strategy"
        
        self._add_perfect_belief(agent_id, "strategy", strategy_desc, 1.0)
        
        # PERFECT VALUATION - use actual limit price instead of inferred price
        # Original always created valuation belief (even with None), we do same but with perfect info
        if hasattr(trader, 'orders') and trader.orders:
            true_valuation = trader.orders[0].price  # Their REAL limit!
            self._add_perfect_belief(agent_id, "valuation", true_valuation, 1.0)
        else:
            # Create valuation belief with None (matching original behavior)
            self._add_perfect_belief(agent_id, "valuation", None, 0.1)
        
        # REMOVED: All extra fields (aggressiveness, inventory, margins, etc.)
        # We only use the original 2 belief types that existed before!
    
    def _add_perfect_belief(self, agent_id: str, belief_type: str, value: Any, confidence: float = 1.0):
        """Add a belief with perfect confidence (avoids duplicates)"""
        # Find existing belief
        for edge_id, edge in self.edges.items():
            if edge.target_node == agent_id and edge.belief_type == belief_type:
                # Update existing
                edge.value = value
                edge.confidence = confidence
                edge.timestamp = self.current_time
                return
        
        # Add new belief
        edge = BeliefEdge(
            edge_id=str(uuid.uuid4()),
            source_node="PERFECT_OBSERVER",  # Mark as perfect info
            target_node=agent_id,
            belief_type=belief_type,
            confidence=confidence,
            value=value,
            timestamp=self.current_time,
            evidence_count=999  # Mark as perfect evidence
        )
        self.edges[edge.edge_id] = edge
        
        
    
    def update_beliefs(self, event: MarketEvent) -> None:
        """
        Update belief graph with PERFECT INFORMATION from market event + trader internals
        """
        self.current_time = event.timestamp
        self.event_history.append(event)
        
        # Ensure the agent exists in the graph with PERFECT info
        if event.agent_id and event.agent_id not in self.nodes:
            self.add_agent(event.agent_id)
        
        # Update with PERFECT information from event + trader internals
        if event.event_type == EventType.BID:
            self._update_perfect_beliefs_from_bid(event)
        elif event.event_type == EventType.ASK:
            self._update_perfect_beliefs_from_ask(event)
        elif event.event_type == EventType.TRADE:
            self._update_perfect_beliefs_from_trade(event)
        elif event.event_type == EventType.CANCEL:
            self._update_perfect_beliefs_from_cancel(event)
        
        # Update asset state with perfect info
        self._update_asset_state()
        
        # Refresh ALL perfect beliefs (no decay, always perfect)
        self._refresh_perfect_beliefs()
        
    def _refresh_perfect_beliefs(self):
        """Refresh all beliefs with current perfect information"""
        for agent_id in list(self.nodes.keys()):
            if agent_id != self.asset_id:  # Skip asset node
                trader = self.traders_dict.get(agent_id)
                if trader:
                    # Update PERFECT beliefs in real-time
                    self._update_perfect_trader_beliefs(agent_id, trader)
                    
    def _update_perfect_trader_beliefs(self, agent_id: str, trader):
        """
        Update ONLY original belief types with current perfect info.
        
        We only update "strategy" and "valuation" - the 2 original belief types.
        """
        # Update perfect strategy with detailed description
        if trader.ttype == "ZIC":
            strategy_desc = "zero_intelligence_constrained_random_trader_that_picks_random_prices_between_system_min_max_but_respects_limit_price_constraints_never_trading_at_loss_purely_random_within_profitable_bounds_no_learning_or_adaptation_just_uniform_random_quote_generation"
        elif trader.ttype == "SHVR":
            strategy_desc = "aggressive_price_improvement_shaver_that_always_tries_to_beat_current_best_price_by_exactly_one_penny_if_bids_exist_quotes_best_bid_plus_1_if_asks_exist_quotes_best_ask_minus_1_but_never_exceeds_own_limit_price_creates_stub_quotes_at_system_extremes_when_no_competition_exists_maximally_competitive_minimal_profit_margin_strategy"
        elif trader.ttype == "SNPR":
            strategy_desc = "time_sensitive_sniping_trader_that_lurks_passively_early_in_session_but_becomes_increasingly_aggressive_as_countdown_timer_approaches_zero_starts_conservative_then_ramps_up_urgency_willing_to_accept_worse_prices_as_time_pressure_mounts_designed_to_capture_last_minute_trading_opportunities"
        elif trader.ttype == "ZIP":
            strategy_desc = "zero_intelligence_plus_adaptive_learning_trader_with_dynamic_profit_margins_that_learns_from_market_feedback_adjusts_bid_sell_margins_based_on_recent_success_failure_uses_momentum_and_learning_rates_to_evolve_strategy_over_time_starts_with_random_margins_then_optimizes_through_reinforcement_learning_from_accepted_rejected_quotes"
        elif trader.ttype == "PT1":
            strategy_desc = "proprietary_buy_and_hold_value_trader_that_waits_5_minutes_for_prices_to_settle_then_buys_when_best_ask_is_below_recent_transaction_price_average_holds_inventory_until_can_sell_at_purchase_price_plus_fixed_profit_margin_long_only_strategy_with_patience_driven_value_investing_approach_only_trades_when_confident_of_profit"
        elif trader.ttype == "PT2":
            strategy_desc = "enhanced_proprietary_buy_and_hold_trader_similar_to_PT1_but_with_refined_parameters_for_bid_percentage_and_profit_margins_also_waits_for_market_settlement_buys_undervalued_assets_based_on_recent_price_history_holds_for_profitable_exit_but_with_different_risk_tolerance_and_patience_thresholds_than_PT1"
        elif trader.ttype == "PRZI":
            strategy_desc = "parameterized_response_zero_intelligence_trader_that_uses_clifford_response_function_to_determine_quote_prices_based_on_mathematical_model_of_market_conditions_adjusts_aggressiveness_based_on_order_book_state_and_internal_parameters_more_sophisticated_than_basic_ZIC_but_still_follows_deterministic_rules_rather_than_learning"
        elif trader.ttype == "GVWY":
            strategy_desc = "giveaway_trader_that_is_even_more_passive_than_zero_intelligence_always_quotes_at_limit_price_essentially_giving_away_profits_to_counterparties_but_never_trading_at_a_loss_minimal_competitiveness_maximum_generosity_while_still_respecting_profitability_constraints_designed_as_market_maker_of_last_resort"
        elif trader.ttype == "BG":
            strategy_desc = "belief_graph_enabled_large_language_model_trader_that_maintains_dynamic_beliefs_about_other_market_participants_strategies_valuations_and_behaviors_uses_AI_reasoning_to_make_trading_decisions_based_on_inferred_competitor_models_and_market_state_representation_this_is_us_the_current_trader_type"
        elif trader.ttype == "LLMHM":
            strategy_desc = "hypothetical_minds_large_language_model_trader_that_generates_hypothetical_scenarios_about_market_participant_behaviors_and_tests_strategies_against_simulated_outcomes_uses_counterfactual_reasoning_and_what_if_analysis_to_inform_trading_decisions_advanced_AI_with_predictive_modeling_capabilities"
        else:
            strategy_desc = "unknown_strategy_type_with_unspecified_business_logic_and_trading_behavior_patterns_may_have_custom_implementation_not_covered_by_standard_BSE_trader_types_proceed_with_caution_and_observe_actual_trading_patterns_to_infer_strategy"
        
        self._update_or_add_belief(agent_id, "strategy", strategy_desc, 1.0)
            
        # Update perfect valuation (original field)
        perfect_valuation = None
        if hasattr(trader, 'orders') and trader.orders:
            perfect_valuation = trader.orders[0].price  # Their REAL limit!
        elif hasattr(trader, 'limit') and trader.limit is not None:
            perfect_valuation = trader.limit  # Some traders may store limit here
        elif hasattr(trader, 'assignment') and trader.assignment is not None:
            perfect_valuation = trader.assignment  # Assignment price is their limit
        
        if perfect_valuation is not None:
            self._update_or_add_belief(agent_id, "valuation", perfect_valuation, 1.0)
        else:
            self._update_or_add_belief(agent_id, "valuation", None, 0.1)
            
        # REMOVED: All other fields (inventory, balance, margins, etc.) 
        # We only use the original 2 belief types!
            
        
    def _update_or_add_belief(self, agent_id: str, belief_type: str, value: Any, confidence: float):
        """Update existing belief or add new one"""
        # Find existing belief
        for edge_id, edge in self.edges.items():
            if edge.target_node == agent_id and edge.belief_type == belief_type:
                # Update existing
                edge.value = value
                edge.confidence = confidence
                edge.timestamp = self.current_time
                return
        
        # Add new belief
        self._add_perfect_belief(agent_id, belief_type, value, confidence)
    
    def _update_perfect_beliefs_from_bid(self, event: MarketEvent) -> None:
        """Update beliefs from bid with PERFECT trader information"""
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_bid_price = event.price
        agent_node.last_activity = event.timestamp
        
        # Get PERFECT information from trader object
        trader = self.traders_dict.get(event.agent_id)
        if trader:
            self._update_perfect_trader_beliefs(event.agent_id, trader)
        
        # Update asset state
        if self.asset_node.current_best_bid is None or event.price > self.asset_node.current_best_bid:
            self.asset_node.current_best_bid = event.price
            
    def _update_perfect_beliefs_from_ask(self, event: MarketEvent) -> None:
        """Update beliefs from ask with PERFECT trader information"""
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_ask_price = event.price
        agent_node.last_activity = event.timestamp
        
        # Get PERFECT information from trader object
        trader = self.traders_dict.get(event.agent_id)
        if trader:
            self._update_perfect_trader_beliefs(event.agent_id, trader)
        
        # Update asset state
        if self.asset_node.current_best_ask is None or event.price < self.asset_node.current_best_ask:
            self.asset_node.current_best_ask = event.price
            
    def _update_perfect_beliefs_from_trade(self, event: MarketEvent) -> None:
        """Update beliefs from trade with PERFECT trader information"""
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_trade_price = event.price
        agent_node.total_trades += 1
        agent_node.total_volume += event.quantity or 1
        agent_node.last_activity = event.timestamp
        
        # Get PERFECT information from trader object
        trader = self.traders_dict.get(event.agent_id)
        if trader:
            self._update_perfect_trader_beliefs(event.agent_id, trader)
        
        # Update asset state with perfect info
        self.asset_node.last_trade_price = event.price
        self.asset_node.volume_traded += event.quantity or 1
        
    def _update_perfect_beliefs_from_cancel(self, event: MarketEvent) -> None:
        """Update beliefs from cancel with PERFECT trader information"""
        if not event.agent_id:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_activity = event.timestamp
        
        # Get PERFECT information from trader object
        trader = self.traders_dict.get(event.agent_id)
        if trader:
            self._update_perfect_trader_beliefs(event.agent_id, trader)

    def _update_beliefs_from_bid(self, event: MarketEvent) -> None:
        """Update beliefs from bid - delegates to perfect version"""
        return self._update_perfect_beliefs_from_bid(event)
    
    def _update_beliefs_from_ask(self, event: MarketEvent) -> None:
        """Update beliefs from ask - delegates to perfect version"""
        return self._update_perfect_beliefs_from_ask(event)
    
    def _update_beliefs_from_trade(self, event: MarketEvent) -> None:
        """Update beliefs from trade - delegates to perfect version"""
        return self._update_perfect_beliefs_from_trade(event)
    
    def _update_beliefs_from_cancel(self, event: MarketEvent) -> None:
        """Update beliefs from cancel - delegates to perfect version"""
        return self._update_perfect_beliefs_from_cancel(event)
    
    def _update_valuation_belief(self, agent_id: str, price: float, action_type: str, high_confidence: bool = False) -> None:
        """Update valuation belief with PERFECT information when available"""
        trader = self.traders_dict.get(agent_id) if self.traders_dict else None
        
        if trader:
            perfect_valuation = None
            if hasattr(trader, 'orders') and trader.orders:
                perfect_valuation = trader.orders[0].price  # Their REAL limit!
            elif hasattr(trader, 'limit') and trader.limit is not None:
                perfect_valuation = trader.limit  # Some traders may store limit here
            elif hasattr(trader, 'assignment') and trader.assignment is not None:
                perfect_valuation = trader.assignment  # Assignment price is their limit
            
            if perfect_valuation is not None:
                self._update_or_add_belief(agent_id, "valuation", perfect_valuation, 1.0)
            else:
                self._update_or_add_belief(agent_id, "valuation", None, 0.1)
        else:
            raise ValueError(f"No trader found for agent {agent_id}")
    
    def _update_strategy_belief(self, agent_id: str, action_type: str, price: float) -> None:
        """Update strategy belief with PERFECT information when available"""
        trader = self.traders_dict.get(agent_id) if self.traders_dict else None
        
        if trader:
            if trader.ttype == "ZIC":
                strategy_desc = "zero_intelligence_constrained_random_trader_that_picks_random_prices_between_system_min_max_but_respects_limit_price_constraints_never_trading_at_loss_purely_random_within_profitable_bounds_no_learning_or_adaptation_just_uniform_random_quote_generation"
            elif trader.ttype == "SHVR":
                strategy_desc = "aggressive_price_improvement_shaver_that_always_tries_to_beat_current_best_price_by_exactly_one_penny_if_bids_exist_quotes_best_bid_plus_1_if_asks_exist_quotes_best_ask_minus_1_but_never_exceeds_own_limit_price_creates_stub_quotes_at_system_extremes_when_no_competition_exists_maximally_competitive_minimal_profit_margin_strategy"
            elif trader.ttype == "SNPR":
                strategy_desc = "time_sensitive_sniping_trader_that_lurks_passively_early_in_session_but_becomes_increasingly_aggressive_as_countdown_timer_approaches_zero_starts_conservative_then_ramps_up_urgency_willing_to_accept_worse_prices_as_time_pressure_mounts_designed_to_capture_last_minute_trading_opportunities"
            elif trader.ttype == "ZIP":
                strategy_desc = "zero_intelligence_plus_adaptive_learning_trader_with_dynamic_profit_margins_that_learns_from_market_feedback_adjusts_bid_sell_margins_based_on_recent_success_failure_uses_momentum_and_learning_rates_to_evolve_strategy_over_time_starts_with_random_margins_then_optimizes_through_reinforcement_learning_from_accepted_rejected_quotes"
            elif trader.ttype == "PT1":
                strategy_desc = "proprietary_buy_and_hold_value_trader_that_waits_5_minutes_for_prices_to_settle_then_buys_when_best_ask_is_below_recent_transaction_price_average_holds_inventory_until_can_sell_at_purchase_price_plus_fixed_profit_margin_long_only_strategy_with_patience_driven_value_investing_approach_only_trades_when_confident_of_profit"
            elif trader.ttype == "PT2":
                strategy_desc = "enhanced_proprietary_buy_and_hold_trader_similar_to_PT1_but_with_refined_parameters_for_bid_percentage_and_profit_margins_also_waits_for_market_settlement_buys_undervalued_assets_based_on_recent_price_history_holds_for_profitable_exit_but_with_different_risk_tolerance_and_patience_thresholds_than_PT1"
            elif trader.ttype == "PRZI":
                strategy_desc = "parameterized_response_zero_intelligence_trader_that_uses_clifford_response_function_to_determine_quote_prices_based_on_mathematical_model_of_market_conditions_adjusts_aggressiveness_based_on_order_book_state_and_internal_parameters_more_sophisticated_than_basic_ZIC_but_still_follows_deterministic_rules_rather_than_learning"
            elif trader.ttype == "GVWY":
                strategy_desc = "giveaway_trader_that_is_even_more_passive_than_zero_intelligence_always_quotes_at_limit_price_essentially_giving_away_profits_to_counterparties_but_never_trading_at_a_loss_minimal_competitiveness_maximum_generosity_while_still_respecting_profitability_constraints_designed_as_market_maker_of_last_resort"
            elif trader.ttype == "BG":
                strategy_desc = "belief_graph_enabled_large_language_model_trader_that_maintains_dynamic_beliefs_about_other_market_participants_strategies_valuations_and_behaviors_uses_AI_reasoning_to_make_trading_decisions_based_on_inferred_competitor_models_and_market_state_representation_this_is_us_the_current_trader_type"
            elif trader.ttype == "LLMHM":
                strategy_desc = "hypothetical_minds_large_language_model_trader_that_generates_hypothetical_scenarios_about_market_participant_behaviors_and_tests_strategies_against_simulated_outcomes_uses_counterfactual_reasoning_and_what_if_analysis_to_inform_trading_decisions_advanced_AI_with_predictive_modeling_capabilities"
            else:
                strategy_desc = "unknown_strategy_type_with_unspecified_business_logic_and_trading_behavior_patterns_may_have_custom_implementation_not_covered_by_standard_BSE_trader_types_proceed_with_caution_and_observe_actual_trading_patterns_to_infer_strategy"
            
            self._update_or_add_belief(agent_id, "strategy", strategy_desc, 1.0)
        else:
            raise ValueError(f"No trader found for agent {agent_id}")
    
    def _update_asset_state(self) -> None:
        """Update the asset node state based on current market conditions"""
        # This would typically be called with actual market data
        # For now, we'll update based on the belief graph state
        
        # Calculate spread if we have both bid and ask
        if (self.asset_node.current_best_bid is not None and 
            self.asset_node.current_best_ask is not None):
            self.asset_node.spread_width = self.asset_node.current_best_ask - self.asset_node.current_best_bid
    
    def _decay_old_beliefs(self) -> None:
        """Decay confidence in old beliefs"""
        current_time = self.current_time
        for edge in self.edges.values():
            time_diff = current_time - edge.timestamp
            if time_diff > 100:  # Decay beliefs older than 100 time units
                decay_factor = self.valuation_decay_rate ** (time_diff / 100)
                edge.confidence *= decay_factor
    
    def query_action(self, agent_id: str, current_market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the belief graph for decision-making.
        
        This function serializes the belief graph and returns it in a format
        suitable for LLM processing.
        """
        # Update asset state with current market data
        if 'best_bid' in current_market_state:
            self.asset_node.current_best_bid = current_market_state['best_bid']
        if 'best_ask' in current_market_state:
            self.asset_node.current_best_ask = current_market_state['best_ask']
        if 'last_trade' in current_market_state:
            self.asset_node.last_trade_price = current_market_state['last_trade']
        
        # Initialize belief graph data structure
        belief_graph_data = {
            'agents': {},
            'beliefs': [],
            'recent_events': [],
            'strategic_insights': {}
        }
        
        # Generate competitor beliefs using existing edges
        competitor_beliefs = {}
        for node_id, node in self.nodes.items():
            if isinstance(node, AgentNode):
                belief_graph_data['agents'][node_id] = node.to_dict()
        
        # Add belief edges
        for edge in self.edges.values():
            belief_graph_data['beliefs'].append(edge.to_dict())
        
        # Add recent events (last 10)
        recent_events = self.event_history[-10:] if len(self.event_history) > 10 else self.event_history
        belief_graph_data['recent_events'] = [event.to_dict() for event in recent_events]
        
        # Add strategic insights
        belief_graph_data['strategic_insights'] = self._generate_strategic_insights(agent_id)
        
        return belief_graph_data
    
    def _generate_strategic_insights(self, agent_id: str) -> Dict[str, Any]:
        """Generate PERFECT strategic insights using ground truth information"""
        insights = {
            'competitors': [],
            'market_opportunities': [],
            'risk_factors': []
        }
        
        # Analyze competitors using ONLY original 2 belief types with PERFECT VALUES
        for node_id, node in self.nodes.items():
            if isinstance(node, AgentNode) and node_id != agent_id:
                # Extract ONLY original belief types with perfect values
                strategy = "unknown"  # Original default
                valuation_estimate = None  # Original field
                valuation_confidence = 0.0  # Original field
                
                for edge in self.edges.values():
                    if edge.target_node == node_id:
                        if edge.belief_type == "strategy":
                            strategy = edge.value  # Perfect: "ZIP", "PT1" instead of "unknown"
                        elif edge.belief_type == "valuation":
                            valuation_estimate = edge.value  # Perfect: actual limit price
                            valuation_confidence = edge.confidence  # Perfect: 1.0 confidence
                        # REMOVED: aggressiveness, predictions, inventory, etc.
                
                competitor_info = {
                    'agent_id': node_id,
                    'strategy': strategy,  # Perfect strategy type
                    'aggressiveness': node.aggressiveness_score,  # Original node field (inferred)
                    'valuation_estimate': valuation_estimate,  # Perfect valuation
                    'confidence': valuation_confidence,  # Perfect confidence 
                    'recent_activity': node.last_activity  # Original node field
                    # REMOVED: perfect_predictions and other extra fields
                }
                insights['competitors'].append(competitor_info)
        
        # Identify market opportunities
        if (self.asset_node.current_best_bid is not None and 
            self.asset_node.current_best_ask is not None):
            spread = self.asset_node.current_best_ask - self.asset_node.current_best_bid
            if spread > 5:  # Arbitrage opportunity
                insights['market_opportunities'].append({
                    'type': 'arbitrage',
                    'spread': spread,
                    'description': f"Large spread of {spread} points"
                })
        
        # Identify risk factors
        if self.asset_node.price_volatility > 0.1:
            insights['risk_factors'].append({
                'type': 'high_volatility',
                'value': self.asset_node.price_volatility,
                'description': "High price volatility detected"
            })
        
        return insights
    
    def to_json(self) -> str:
        """Serialize the belief graph to JSON"""
        graph_data = {
            'graph_id': self.graph_id,
            'asset_id': self.asset_id,
            'current_time': self.current_time,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            'event_history': [event.to_dict() for event in self.event_history[-50:]]  # Last 50 events
        }
        return json.dumps(graph_data, indent=2)
    
    def from_json(self, json_str: str) -> None:
        """Deserialize the belief graph from JSON"""
        data = json.loads(json_str)
        self.graph_id = data['graph_id']
        self.asset_id = data['asset_id']
        self.current_time = data['current_time']
        
        # Reconstruct nodes
        self.nodes.clear()
        for node_id, node_data in data['nodes'].items():
            if node_data['node_type'] == NodeType.AGENT.value:
                self.nodes[node_id] = AgentNode(**node_data)
            elif node_data['node_type'] == NodeType.ASSET.value:
                self.nodes[node_id] = AssetNode(**node_data)
        
        # Reconstruct edges
        self.edges.clear()
        for edge_id, edge_data in data['edges'].items():
            self.edges[edge_id] = BeliefEdge(**edge_data)
        
        # Reconstruct event history
        self.event_history = [MarketEvent(**event_data) for event_data in data['event_history']]
    
    def get_agent_beliefs(self, agent_id: str) -> Dict[str, Any]:
        """Get all beliefs about a specific agent"""
        beliefs = {}
        
        # First get the raw belief edges
        for edge in self.edges.values():
            if edge.target_node == agent_id:
                if edge.belief_type not in beliefs:
                    beliefs[edge.belief_type] = []
                beliefs[edge.belief_type].append(edge.to_dict())
        
        # Extract valuation estimate if available
        valuation_estimate = None
        if agent_id in self.nodes:
            agent_node = self.nodes[agent_id]
            if hasattr(agent_node, 'inferred_valuation'):
                valuation_estimate = agent_node.inferred_valuation
            
            # Also try to get from valuation edge
            for edge in self.edges.values():
                if (edge.target_node == agent_id and 
                    edge.belief_type == "valuation" and 
                    edge.value and isinstance(edge.value, dict)):
                    possible_vals = edge.value.get("possible_valuations", [])
                    if possible_vals:
                        # Use median of possible valuations as estimate
                        valuation_estimate = sorted(possible_vals)[len(possible_vals)//2]
                        break
        
        # Extract strategy type if available
        strategy_type = None
        if agent_id in self.nodes:
            agent_node = self.nodes[agent_id]
            if hasattr(agent_node, 'strategy_type'):
                strategy_type = agent_node.strategy_type
        
        # Add simplified summary
        beliefs['valuation_estimate'] = valuation_estimate
        beliefs['strategy_type'] = strategy_type
        
        return beliefs
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get a summary of the current market state"""
        return {
            'asset_state': self.asset_node.to_dict(),
            'active_agents': len([n for n in self.nodes.values() if isinstance(n, AgentNode)]),
            'total_beliefs': len(self.edges),
            'recent_events': len(self.event_history),
            'current_time': self.current_time
        }

