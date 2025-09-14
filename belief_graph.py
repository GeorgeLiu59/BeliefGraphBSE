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
        print(f"[BG-DEBUG] _update_beliefs_from_trade called for agent {event.agent_id} at price {event.price}")
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
            print(f"[BG-TRADE-AGGR] Agent {event.agent_id}: price={event.price}, last_price={self.asset_node.last_trade_price}, ratio={price_ratio:.3f}")
            
            # Buyer perspective (assuming agent_id is buyer in BSE)
            if price_ratio > 1.01:  # Paid >1% more than last trade
                agent_node.aggressiveness_score = min(1.0, agent_node.aggressiveness_score + 0.3)
                print(f"[BG-TRADE-AGGR] AGGRESSIVE buyer: paid {price_ratio-1:.1%} more")
            elif price_ratio < 0.99:  # Paid <1% less than last trade  
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.2)
                print(f"[BG-TRADE-AGGR] PASSIVE buyer: paid {1-price_ratio:.1%} less")
            else:
                # Near market price - slight shift toward passive
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.05)
                print(f"[BG-TRADE-AGGR] NEUTRAL trade")
        else:
            # First trade - random initial aggressiveness
            import random
            agent_node.aggressiveness_score = random.uniform(-0.3, 0.3)
            print(f"[BG-TRADE-AGGR] First trade, random initial aggr={agent_node.aggressiveness_score:.3f}")
        
        print(f"[BG-TRADE-AGGR] Agent {event.agent_id} aggr: {old_aggr:.3f} -> {agent_node.aggressiveness_score:.3f}")
        
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
                    print(f"[BG-TRADE-AGGR] AGGRESSIVE seller {event.counterparty_id}: sold {1-price_ratio:.1%} below")
                elif price_ratio > 1.01:  # Sold >1% above last trade
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.2)
                    print(f"[BG-TRADE-AGGR] PASSIVE seller {event.counterparty_id}: sold {price_ratio-1:.1%} above")
                else:
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.05)
            else:
                # First trade - random initial
                import random
                seller_node.aggressiveness_score = random.uniform(-0.3, 0.3)
                
            print(f"[BG-TRADE-AGGR] Seller {event.counterparty_id} aggr: {old_seller_aggr:.3f} -> {seller_node.aggressiveness_score:.3f}")
            
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
        print(f"[BG-STRAT-START] Updating strategy for {agent_id}, action={action_type}, price={price}")
        
        # Find existing strategy edge
        strategy_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "strategy"):
                strategy_edge = edge
                break
        
        if strategy_edge is None:
            print(f"[BG-STRAT-ERROR] No strategy edge found for {agent_id}")
            return
        
        # Simple strategy classification based on behavior patterns
        agent_node = self.nodes[agent_id]
        
        print(f"[BG-STRAT-INFO] Agent {agent_id}: trades={agent_node.total_trades}, aggr_score={agent_node.aggressiveness_score:.3f}, current_strategy={agent_node.strategy_type}")
        
        # Always classify based on aggressiveness, even if no trades yet
        old_strategy = agent_node.strategy_type
        
        # Use more sensitive thresholds
        if agent_node.aggressiveness_score > 0.15:
            strategy = "aggressive"
        elif agent_node.aggressiveness_score < -0.15:
            strategy = "passive"
        else:
            strategy = "neutral"
        
        print(f"[BG-STRAT-CLASSIFY] Agent {agent_id}: aggr={agent_node.aggressiveness_score:.3f} -> strategy={strategy} (was {old_strategy})")
        
        strategy_edge.value = strategy
        strategy_edge.confidence = min(self.max_confidence, strategy_edge.confidence + 0.1)
        strategy_edge.timestamp = self.current_time
        strategy_edge.evidence_count += 1
        
        # Sync the agent node's strategy_type field
        agent_node.strategy_type = strategy
        
        print(f"[BG-STRAT-END] Agent {agent_id} strategy updated to {strategy} (confidence={strategy_edge.confidence:.2f})")
    
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



class GraphVar1:
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
        print(f"[BG-DEBUG] _update_beliefs_from_trade called for agent {event.agent_id} at price {event.price}")
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
            print(f"[BG-TRADE-AGGR] Agent {event.agent_id}: price={event.price}, last_price={self.asset_node.last_trade_price}, ratio={price_ratio:.3f}")
            
            # Buyer perspective (assuming agent_id is buyer in BSE)
            if price_ratio > 1.01:  # Paid >1% more than last trade
                agent_node.aggressiveness_score = min(1.0, agent_node.aggressiveness_score + 0.3)
                print(f"[BG-TRADE-AGGR] AGGRESSIVE buyer: paid {price_ratio-1:.1%} more")
            elif price_ratio < 0.99:  # Paid <1% less than last trade  
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.2)
                print(f"[BG-TRADE-AGGR] PASSIVE buyer: paid {1-price_ratio:.1%} less")
            else:
                # Near market price - slight shift toward passive
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.05)
                print(f"[BG-TRADE-AGGR] NEUTRAL trade")
        else:
            # First trade - random initial aggressiveness
            import random
            agent_node.aggressiveness_score = random.uniform(-0.3, 0.3)
            print(f"[BG-TRADE-AGGR] First trade, random initial aggr={agent_node.aggressiveness_score:.3f}")
        
        print(f"[BG-TRADE-AGGR] Agent {event.agent_id} aggr: {old_aggr:.3f} -> {agent_node.aggressiveness_score:.3f}")
        
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
                    print(f"[BG-TRADE-AGGR] AGGRESSIVE seller {event.counterparty_id}: sold {1-price_ratio:.1%} below")
                elif price_ratio > 1.01:  # Sold >1% above last trade
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.2)
                    print(f"[BG-TRADE-AGGR] PASSIVE seller {event.counterparty_id}: sold {price_ratio-1:.1%} above")
                else:
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.05)
            else:
                # First trade - random initial
                import random
                seller_node.aggressiveness_score = random.uniform(-0.3, 0.3)
                
            print(f"[BG-TRADE-AGGR] Seller {event.counterparty_id} aggr: {old_seller_aggr:.3f} -> {seller_node.aggressiveness_score:.3f}")
            
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
        print(f"[BG-STRAT-START] Updating strategy for {agent_id}, action={action_type}, price={price}")
        
        # Find existing strategy edge
        strategy_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "strategy"):
                strategy_edge = edge
                break
        
        if strategy_edge is None:
            print(f"[BG-STRAT-ERROR] No strategy edge found for {agent_id}")
            return
        
        # Simple strategy classification based on behavior patterns
        agent_node = self.nodes[agent_id]
        
        print(f"[BG-STRAT-INFO] Agent {agent_id}: trades={agent_node.total_trades}, aggr_score={agent_node.aggressiveness_score:.3f}, current_strategy={agent_node.strategy_type}")
        
        # Always classify based on aggressiveness, even if no trades yet
        old_strategy = agent_node.strategy_type
        
        # Use more sensitive thresholds
        if agent_node.aggressiveness_score > 0.15:
            strategy = "aggressive"
        elif agent_node.aggressiveness_score < -0.15:
            strategy = "passive"
        else:
            strategy = "neutral"
        
        print(f"[BG-STRAT-CLASSIFY] Agent {agent_id}: aggr={agent_node.aggressiveness_score:.3f} -> strategy={strategy} (was {old_strategy})")
        
        strategy_edge.value = strategy
        strategy_edge.confidence = min(self.max_confidence, strategy_edge.confidence + 0.1)
        strategy_edge.timestamp = self.current_time
        strategy_edge.evidence_count += 1
        
        # Sync the agent node's strategy_type field
        agent_node.strategy_type = strategy
        
        print(f"[BG-STRAT-END] Agent {agent_id} strategy updated to {strategy} (confidence={strategy_edge.confidence:.2f})")
    
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


class GraphVar2:
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
        print(f"[BG-DEBUG] _update_beliefs_from_trade called for agent {event.agent_id} at price {event.price}")
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
            print(f"[BG-TRADE-AGGR] Agent {event.agent_id}: price={event.price}, last_price={self.asset_node.last_trade_price}, ratio={price_ratio:.3f}")
            
            # Buyer perspective (assuming agent_id is buyer in BSE)
            if price_ratio > 1.01:  # Paid >1% more than last trade
                agent_node.aggressiveness_score = min(1.0, agent_node.aggressiveness_score + 0.3)
                print(f"[BG-TRADE-AGGR] AGGRESSIVE buyer: paid {price_ratio-1:.1%} more")
            elif price_ratio < 0.99:  # Paid <1% less than last trade  
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.2)
                print(f"[BG-TRADE-AGGR] PASSIVE buyer: paid {1-price_ratio:.1%} less")
            else:
                # Near market price - slight shift toward passive
                agent_node.aggressiveness_score = max(-1.0, agent_node.aggressiveness_score - 0.05)
                print(f"[BG-TRADE-AGGR] NEUTRAL trade")
        else:
            # First trade - random initial aggressiveness
            import random
            agent_node.aggressiveness_score = random.uniform(-0.3, 0.3)
            print(f"[BG-TRADE-AGGR] First trade, random initial aggr={agent_node.aggressiveness_score:.3f}")
        
        print(f"[BG-TRADE-AGGR] Agent {event.agent_id} aggr: {old_aggr:.3f} -> {agent_node.aggressiveness_score:.3f}")
        
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
                    print(f"[BG-TRADE-AGGR] AGGRESSIVE seller {event.counterparty_id}: sold {1-price_ratio:.1%} below")
                elif price_ratio > 1.01:  # Sold >1% above last trade
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.2)
                    print(f"[BG-TRADE-AGGR] PASSIVE seller {event.counterparty_id}: sold {price_ratio-1:.1%} above")
                else:
                    seller_node.aggressiveness_score = max(-1.0, seller_node.aggressiveness_score - 0.05)
            else:
                # First trade - random initial
                import random
                seller_node.aggressiveness_score = random.uniform(-0.3, 0.3)
                
            print(f"[BG-TRADE-AGGR] Seller {event.counterparty_id} aggr: {old_seller_aggr:.3f} -> {seller_node.aggressiveness_score:.3f}")
            
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
        print(f"[BG-STRAT-START] Updating strategy for {agent_id}, action={action_type}, price={price}")
        
        # Find existing strategy edge
        strategy_edge = None
        for edge in self.edges.values():
            if (edge.source_node == self.asset_id and 
                edge.target_node == agent_id and 
                edge.belief_type == "strategy"):
                strategy_edge = edge
                break
        
        if strategy_edge is None:
            print(f"[BG-STRAT-ERROR] No strategy edge found for {agent_id}")
            return
        
        # Simple strategy classification based on behavior patterns
        agent_node = self.nodes[agent_id]
        
        print(f"[BG-STRAT-INFO] Agent {agent_id}: trades={agent_node.total_trades}, aggr_score={agent_node.aggressiveness_score:.3f}, current_strategy={agent_node.strategy_type}")
        
        # Always classify based on aggressiveness, even if no trades yet
        old_strategy = agent_node.strategy_type
        
        # Use more sensitive thresholds
        if agent_node.aggressiveness_score > 0.15:
            strategy = "aggressive"
        elif agent_node.aggressiveness_score < -0.15:
            strategy = "passive"
        else:
            strategy = "neutral"
        
        print(f"[BG-STRAT-CLASSIFY] Agent {agent_id}: aggr={agent_node.aggressiveness_score:.3f} -> strategy={strategy} (was {old_strategy})")
        
        strategy_edge.value = strategy
        strategy_edge.confidence = min(self.max_confidence, strategy_edge.confidence + 0.1)
        strategy_edge.timestamp = self.current_time
        strategy_edge.evidence_count += 1
        
        # Sync the agent node's strategy_type field
        agent_node.strategy_type = strategy
        
        print(f"[BG-STRAT-END] Agent {agent_id} strategy updated to {strategy} (confidence={strategy_edge.confidence:.2f})")
    
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

