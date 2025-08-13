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
        
        # Update aggressiveness score
        if self.asset_node.current_best_bid is not None:
            if event.price > self.asset_node.current_best_bid:
                agent_node.aggressiveness_score = min(1.0, agent_node.aggressiveness_score + 0.1)
            else:
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.05)
        
        # Update strategy belief
        self._update_strategy_belief(event.agent_id, "bid", event.price)
    
    def _update_beliefs_from_ask(self, event: MarketEvent) -> None:
        """Update beliefs based on an ask event"""
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_ask_price = event.price
        agent_node.last_activity = event.timestamp
        
        # Update valuation belief
        self._update_valuation_belief(event.agent_id, event.price, "ask")
        
        # Update aggressiveness score
        if self.asset_node.current_best_ask is not None:
            if event.price < self.asset_node.current_best_ask:
                agent_node.aggressiveness_score = min(1.0, agent_node.aggressiveness_score + 0.1)
            else:
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.05)
        
        # Update strategy belief
        self._update_strategy_belief(event.agent_id, "ask", event.price)
    
    def _update_beliefs_from_trade(self, event: MarketEvent) -> None:
        """Update beliefs based on a trade event"""
        if not event.agent_id or event.price is None:
            return
            
        agent_node = self.nodes[event.agent_id]
        agent_node.last_trade_price = event.price
        agent_node.total_trades += 1
        agent_node.total_volume += event.quantity or 1
        agent_node.last_activity = event.timestamp
        
        # Update valuation belief with high confidence (actual trade)
        self._update_valuation_belief(event.agent_id, event.price, "trade", high_confidence=True)
        
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
    
    def _update_strategy_belief(self, agent_id: str, action_type: str, price: float) -> None:
        """Update the belief about an agent's strategy"""
        # Find existing strategy edge
        strategy_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "strategy"):
                strategy_edge = edge
                break
        
        if strategy_edge is None:
            return
        
        # Simple strategy classification based on behavior patterns
        agent_node = self.nodes[agent_id]
        
        if agent_node.total_trades > 5:
            # Classify based on trading patterns
            if agent_node.aggressiveness_score > 0.5:
                strategy = "aggressive"
            elif agent_node.aggressiveness_score < -0.5:
                strategy = "passive"
            else:
                strategy = "neutral"
            
            strategy_edge.value = strategy
            strategy_edge.confidence = min(self.max_confidence, strategy_edge.confidence + 0.1)
            strategy_edge.timestamp = self.current_time
            strategy_edge.evidence_count += 1
    
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
    
    def get_agent_beliefs(self, agent_id: str) -> Dict[str, Any]:
        """Get all beliefs about a specific agent"""
        beliefs = {}
        for edge in self.edges.values():
            if edge.target_node == agent_id:
                if edge.belief_type not in beliefs:
                    beliefs[edge.belief_type] = []
                beliefs[edge.belief_type].append(edge.to_dict())
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

