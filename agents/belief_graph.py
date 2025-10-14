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

# Default loggers for GraphVar classes (will be overridden by trader's logger)
gv1_logger = logging.getLogger(__name__)
gv2_logger = logging.getLogger(__name__)
gv3_logger = logging.getLogger(__name__)
bg_logger = logging.getLogger(__name__)


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

    @property
    def agents(self):
        """Return list of agent IDs (excluding asset node)"""
        return [node_id for node_id in self.nodes.keys() if node_id != self.asset_id]

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

        # Update agent node
        agent_node.last_trade_price = event.price
        agent_node.total_trades += 1
        agent_node.total_volume += event.quantity or 1
        agent_node.last_activity = event.timestamp

        # Update asset node
        self.asset_node.last_trade_price = event.price
        self.asset_node.volume_traded += event.quantity or 1

        # Update valuation belief with high confidence (actual trade)
        self._update_valuation_belief(event.agent_id, event.price, "trade", high_confidence=True)

        # Update strategy belief based on trade
        self._update_strategy_belief(event.agent_id, "trade", event.price)

        # Also update counterparty if available
        if event.counterparty_id and event.counterparty_id in self.nodes:
            seller_node = self.nodes[event.counterparty_id]
            seller_node.last_trade_price = event.price
            seller_node.total_trades += 1
            seller_node.total_volume += event.quantity or 1

            # Update seller strategy
            self._update_valuation_belief(event.counterparty_id, event.price, "trade", high_confidence=True)
            self._update_strategy_belief(event.counterparty_id, "trade", event.price)
    
    def _update_beliefs_from_cancel(self, event: MarketEvent) -> None:
        """Update beliefs based on a cancel event"""
        if not event.agent_id:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_activity = event.timestamp

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

        # Update timestamp and confidence
        strategy_edge.confidence = min(self.max_confidence, strategy_edge.confidence + 0.1)
        strategy_edge.timestamp = self.current_time
        strategy_edge.evidence_count += 1

        bg_logger.debug(f"[BG-STRAT-END] Agent {agent_id} strategy updated (confidence={strategy_edge.confidence:.2f})")
    
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

        # Store all AI-designed attributes as belief edges
        for attr_name, attr_value in attributes.items():
            if attr_name == 'reasoning':
                continue

            belief_edge = BeliefEdge(
                edge_id=str(uuid.uuid4()),
                source_node=self.asset_id,
                target_node=agent_id,
                belief_type=attr_name,
                confidence=0.9,
                value=attr_value,
                timestamp=self.current_time
            )
            self.edges[belief_edge.edge_id] = belief_edge

        agent_node.last_activity = self.current_time
    
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

        # Extract from edges (works for all graph types)
        if 'strategy' in beliefs and beliefs['strategy']:
            strategy_edge = beliefs['strategy'][0]
            strategy_type = strategy_edge.get('value', strategy_type)

        # For valuation, also try direct edge value
        if 'valuation' in beliefs and beliefs['valuation']:
            val_edge = beliefs['valuation'][0]
            val_value = val_edge.get('value')
            if val_value is not None and not isinstance(val_value, dict):
                valuation_estimate = val_value

        # Add simplified summary
        beliefs['valuation_estimate'] = valuation_estimate
        beliefs['strategy_type'] = strategy_type

        return beliefs


    def get_beliefs(self, agent_id: str) -> str:
        """Get beliefs about an agent (helper method)"""
        beliefs = self.get_agent_beliefs(agent_id)
        return str(beliefs)

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
    
    def __init__(self, asset_id: str = "DEFAULT_ASSET", model=None, logger=None):
        self.graph_id = str(uuid.uuid4())
        self.asset_id = asset_id
        self.nodes: Dict[str, AgentNode | AssetNode] = {}
        self.edges: Dict[str, BeliefEdge] = {}
        self.event_history: List[MarketEvent] = []
        self.current_time = 0.0

        # LLM model and logger for dynamic belief inference
        self.model = model
        self.logger = logger if logger else gv1_logger

        # Initialize the asset node
        self.asset_node = AssetNode(asset_id=asset_id)
        self.nodes[asset_id] = self.asset_node

        # Belief update parameters
        self.valuation_decay_rate = 0.95
        self.confidence_boost = 0.1
        self.max_confidence = 0.95

        # Store discrete belief sets per agent
        self.discrete_beliefs: Dict[str, Dict[str, Any]] = {}

    def add_agent(self, agent_id: str) -> None:
        """Add a new agent to the belief graph"""
        if agent_id not in self.nodes:
            agent_node = AgentNode(agent_id=agent_id)
            self.nodes[agent_id] = agent_node

            # Initialize discrete beliefs storage
            self.discrete_beliefs[agent_id] = {
                'confidence': 0.1,
                'reasoning': 'Initial state - no observations yet'
            }

            # Add initial beliefs about this agent
            self._add_initial_beliefs(agent_id)

    @property
    def agents(self):
        """Return list of agent IDs (excluding asset node)"""
        return [node_id for node_id in self.nodes.keys() if node_id != self.asset_id]

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

    def _build_agent_history(self, agent_id: str) -> Dict[str, Any]:
        """Build trading history for an agent"""
        if agent_id not in self.nodes:
            return {}

        agent_node = self.nodes[agent_id]
        agent_events = [e for e in self.event_history[-20:] if e.agent_id == agent_id]

        return {
            'total_trades': agent_node.total_trades,
            'total_volume': agent_node.total_volume,
            'last_bid_price': agent_node.last_bid_price,
            'last_ask_price': agent_node.last_ask_price,
            'last_trade_price': agent_node.last_trade_price,
            'last_activity': agent_node.last_activity,
            'recent_events': [
                {
                    'event_type': e.event_type.value,
                    'price': e.price,
                    'quantity': e.quantity,
                    'timestamp': e.timestamp
                } for e in agent_events
            ]
        }

    def _build_market_state(self) -> Dict[str, Any]:
        """Build current market state summary"""
        return {
            'current_best_bid': self.asset_node.current_best_bid,
            'current_best_ask': self.asset_node.current_best_ask,
            'last_trade_price': self.asset_node.last_trade_price,
            'spread_width': self.asset_node.spread_width,
            'volume_traded': self.asset_node.volume_traded,
            'current_time': self.current_time,
            'total_agents': len([n for n in self.nodes.values() if isinstance(n, AgentNode)])
        }

    def _infer_discrete_beliefs_from_event(self, event: MarketEvent) -> None:
        """Infer discrete beliefs using LLM based on observed event"""
        if not self.model or not event.agent_id:
            return

        agent_history = self._build_agent_history(event.agent_id)
        market_state = self._build_market_state()
        current_beliefs = self.discrete_beliefs[event.agent_id]

        from .unified_prompts import AdaptiveAttributePrompts, PromptParser

        prompt = AdaptiveAttributePrompts.infer_discrete_beliefs_prompt(
            event.agent_id,
            event,
            agent_history,
            market_state,
            current_beliefs
        )

        self.logger.info("=== DISCRETE BELIEF INFERENCE PROMPT ===")
        self.logger.info("="*80)
        self.logger.info(prompt)
        self.logger.info("="*80)

        response = self.model.generate_content(
            prompt,
            generation_config=self.model._generation_config
        )

        self.logger.info("=== DISCRETE BELIEF INFERENCE RESPONSE ===")
        self.logger.info(response.text.strip())
        self.logger.info("="*80)

        new_beliefs = PromptParser.parse_discrete_beliefs(response.text)
        old_beliefs = self.discrete_beliefs[event.agent_id].copy()

        self.discrete_beliefs[event.agent_id] = new_beliefs

        # Synchronize edges with discrete beliefs
        import uuid
        confidence = new_beliefs.get('confidence', 0.5)

        for belief_type, belief_value in new_beliefs.items():
            if belief_type in ['confidence', 'reasoning']:
                continue

            # Find and update or create edge for this belief type
            edge_found = False
            for edge in self.edges.values():
                if edge.target_node == event.agent_id and edge.belief_type == belief_type:
                    edge.value = belief_value
                    edge.confidence = confidence
                    edge.timestamp = event.timestamp
                    edge_found = True
                    break

            # Create new edge if not found
            if not edge_found:
                new_edge = BeliefEdge(
                    edge_id=str(uuid.uuid4()),
                    source_node=self.asset_id,
                    target_node=event.agent_id,
                    belief_type=belief_type,
                    confidence=confidence,
                    value=belief_value,
                    timestamp=event.timestamp,
                    evidence_count=1
                )
                self.edges[new_edge.edge_id] = new_edge

        self.logger.info(f"=== DISCRETE BELIEFS UPDATED FOR {event.agent_id} ===")
        self.logger.info(f"OLD: {old_beliefs}")
        self.logger.info(f"NEW: {new_beliefs}")
        self.logger.info("="*80)

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
        """Update beliefs from bid event (LLM-based inference)"""
        if not event.agent_id or event.price is None:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_bid_price = event.price
        agent_node.last_activity = event.timestamp

        # LLM-based discrete belief inference
        self._infer_discrete_beliefs_from_event(event)

    def _update_beliefs_from_ask(self, event: MarketEvent) -> None:
        """Update beliefs from ask event (LLM-based inference)"""
        if not event.agent_id or event.price is None:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_ask_price = event.price
        agent_node.last_activity = event.timestamp

        # LLM-based discrete belief inference
        self._infer_discrete_beliefs_from_event(event)

    def _update_beliefs_from_trade(self, event: MarketEvent) -> None:
        """Update beliefs from trade event (LLM-based inference)"""
        if not event.agent_id or event.price is None:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_trade_price = event.price
        agent_node.total_trades += 1
        agent_node.total_volume += event.quantity or 1
        agent_node.last_activity = event.timestamp

        # Update asset state
        self.asset_node.last_trade_price = event.price
        self.asset_node.volume_traded += event.quantity or 1

        # LLM-based discrete belief inference
        self._infer_discrete_beliefs_from_event(event)

    def _update_beliefs_from_cancel(self, event: MarketEvent) -> None:
        """Update beliefs from cancel event (LLM-based inference)"""
        if not event.agent_id:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_activity = event.timestamp

        # LLM-based discrete belief inference
        self._infer_discrete_beliefs_from_event(event)

    
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
                    observed_behavior = f"bidding_at_{node.last_bid_price}"
                elif node.last_ask_price:
                    observed_behavior = f"asking_at_{node.last_ask_price}"
                elif node.last_trade_price:
                    if node.total_trades >= 3:
                        observed_behavior = f"frequent_trading_at_{node.last_trade_price}"
                    else:
                        observed_behavior = f"trading_at_{node.last_trade_price}"
                
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
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Market_Direction"] = {
                                "actual_behavior_I_observe": f"observed_trading_activity",
                                f"{node_id.lower()}_self_belief": {
                                    "possible_directions": edge.value["possible_directions"]
                                }
                            }
                        
                        elif edge.belief_type == "desperation_level" and "possible_desperation" in edge.value:
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Desperation_Level"] = {
                                "actual_behavior_I_observe": f"observed_order_pattern",
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
            'discrete_beliefs': {
                agent_id: beliefs
                for agent_id, beliefs in self.discrete_beliefs.items()
            },
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

        # Reconstruct discrete beliefs
        if 'discrete_beliefs' in data:
            self.discrete_beliefs = data['discrete_beliefs']

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

        # Extract from edges (works for all graph types)
        if 'strategy' in beliefs and beliefs['strategy']:
            strategy_edge = beliefs['strategy'][0]
            strategy_type = strategy_edge.get('value', strategy_type)

        # For valuation, also try direct edge value
        if 'valuation' in beliefs and beliefs['valuation']:
            val_edge = beliefs['valuation'][0]
            val_value = val_edge.get('value')
            if val_value is not None and not isinstance(val_value, dict):
                valuation_estimate = val_value

        # Add simplified summary
        beliefs['valuation_estimate'] = valuation_estimate
        beliefs['strategy_type'] = strategy_type

        return beliefs


    def get_beliefs(self, agent_id: str) -> str:
        """Get beliefs about an agent (helper method)"""
        beliefs = self.get_agent_beliefs(agent_id)
        return str(beliefs)

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
    Main belief graph class for managing agent beliefs and market state using probabilistic distributions.
    
    The belief graph maintains:
    - Nodes for each agent and the traded asset
    - Edges representing probability distributions about other agents' valuations and strategies
    - Bayesian updates based on market events
    - Query interface for decision-making
    """
    
    def __init__(self, asset_id: str = "DEFAULT_ASSET", model=None, logger=None):
        self.graph_id = str(uuid.uuid4())
        self.asset_id = asset_id
        self.nodes: Dict[str, AgentNode | AssetNode] = {}
        self.edges: Dict[str, BeliefEdge] = {}
        self.event_history: List[MarketEvent] = []
        self.current_time = 0.0

        # LLM model and logger for dynamic belief inference
        self.model = model
        self.logger = logger if logger else gv2_logger

        # Initialize the asset node
        self.asset_node = AssetNode(asset_id=asset_id)
        self.nodes[asset_id] = self.asset_node

        # Belief update parameters - CONFIGURABLE!
        self.valuation_decay_rate = 0.95
        self.confidence_boost = 0.1
        self.max_confidence = 0.95

        # GraphVar2 Bayesian update parameters - configurable likelihoods
        self.bid_support_likelihood = 0.8
        self.bid_contradict_likelihood = 0.2
        self.ask_support_likelihood = 0.8
        self.ask_contradict_likelihood = 0.2
        self.trade_close_likelihood = 0.9
        self.trade_medium_likelihood = 0.6
        self.trade_far_likelihood = 0.1
        self.trade_close_distance = 5
        self.trade_medium_distance = 10

        # Store probabilistic belief distributions per agent
        self.probabilistic_beliefs: Dict[str, Dict[str, Any]] = {}
        
    def add_agent(self, agent_id: str) -> None:
        """Add a new agent to the belief graph"""
        if agent_id not in self.nodes:
            agent_node = AgentNode(agent_id=agent_id)
            self.nodes[agent_id] = agent_node

            # Add initial beliefs about this agent
            self._add_initial_beliefs(agent_id)

    @property
    def agents(self):
        """Return list of agent IDs (excluding asset node)"""
        return [node_id for node_id in self.nodes.keys() if node_id != self.asset_id]

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

    def _build_agent_history(self, agent_id: str) -> Dict[str, Any]:
        """Build agent's action history for LLM context"""
        agent_node = self.nodes[agent_id]
        
        # Extract recent events for this agent
        agent_events = [e for e in self.event_history[-20:] if e.agent_id == agent_id]
        
        return {
            'total_trades': agent_node.total_trades,
            'total_volume': agent_node.total_volume,
            'last_bid_price': agent_node.last_bid_price,
            'last_ask_price': agent_node.last_ask_price,
            'last_trade_price': agent_node.last_trade_price,
            'last_activity': agent_node.last_activity,
            'recent_events': [
                {
                    'event_type': e.event_type.value,
                    'price': e.price,
                    'quantity': e.quantity,
                    'timestamp': e.timestamp
                } for e in agent_events
            ]
        }

    def _build_market_state(self) -> Dict[str, Any]:
        """Build current market state for LLM context"""
        return {
            'current_best_bid': self.asset_node.current_best_bid,
            'current_best_ask': self.asset_node.current_best_ask,
            'last_trade_price': self.asset_node.last_trade_price,
            'spread_width': self.asset_node.spread_width,
            'volatility': self.asset_node.price_volatility,
            'volume_traded': self.asset_node.volume_traded,
            'total_agents': len([n for n in self.nodes.values() if isinstance(n, AgentNode)])
        }

    def _infer_probabilistic_beliefs_from_event(self, event: MarketEvent) -> None:
        """Infer probabilistic beliefs using LLM based on observed event"""
        if not self.model or not event.agent_id:
            return

        agent_history = self._build_agent_history(event.agent_id)
        market_state = self._build_market_state()
        current_beliefs = self.probabilistic_beliefs.get(event.agent_id, {})

        from .unified_prompts import AdaptiveAttributePrompts, PromptParser

        prompt = AdaptiveAttributePrompts.infer_probabilistic_beliefs_prompt(
            event.agent_id,
            event,
            agent_history,
            market_state,
            current_beliefs
        )

        self.logger.info("=== PROBABILISTIC BELIEF INFERENCE PROMPT ===")
        self.logger.info("="*80)
        self.logger.info(prompt)
        self.logger.info("="*80)

        response = self.model.generate_content(
            prompt,
            generation_config=self.model._generation_config
        )

        self.logger.info("=== PROBABILISTIC BELIEF INFERENCE RESPONSE ===")
        self.logger.info(response.text.strip())
        self.logger.info("="*80)

        new_beliefs = PromptParser.parse_probabilistic_beliefs(response.text)
        old_beliefs = self.probabilistic_beliefs.get(event.agent_id, {}).copy()

        self.probabilistic_beliefs[event.agent_id] = new_beliefs

        # Synchronize edges with probabilistic beliefs
        import uuid
        confidence = new_beliefs.get('confidence', 0.5)

        for belief_type, belief_value in new_beliefs.items():
            if belief_type in ['confidence', 'reasoning']:
                continue

            # Find and update or create edge for this belief type
            edge_found = False
            for edge in self.edges.values():
                if edge.target_node == event.agent_id and edge.belief_type == belief_type:
                    edge.value = belief_value
                    edge.confidence = confidence
                    edge.timestamp = event.timestamp
                    edge_found = True
                    break

            # Create new edge if not found
            if not edge_found:
                new_edge = BeliefEdge(
                    edge_id=str(uuid.uuid4()),
                    source_node=self.asset_id,
                    target_node=event.agent_id,
                    belief_type=belief_type,
                    confidence=confidence,
                    value=belief_value,
                    timestamp=event.timestamp,
                    evidence_count=1
                )
                self.edges[new_edge.edge_id] = new_edge

        self.logger.info(f"=== PROBABILISTIC BELIEFS UPDATED FOR {event.agent_id} ===")
        self.logger.info(f"OLD: {old_beliefs}")
        self.logger.info(f"NEW: {new_beliefs}")
        self.logger.info("="*80)

    def _update_beliefs_from_bid(self, event: MarketEvent) -> None:
        """Update beliefs from bid event (LLM-based inference)"""
        if not event.agent_id or event.price is None:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_bid_price = event.price
        agent_node.last_activity = event.timestamp

        # LLM-based probabilistic belief inference
        self._infer_probabilistic_beliefs_from_event(event)

    def _update_beliefs_from_ask(self, event: MarketEvent) -> None:
        """Update beliefs from ask event (LLM-based inference)"""
        if not event.agent_id or event.price is None:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_ask_price = event.price
        agent_node.last_activity = event.timestamp

        # LLM-based probabilistic belief inference
        self._infer_probabilistic_beliefs_from_event(event)

    def _update_beliefs_from_trade(self, event: MarketEvent) -> None:
        """Update beliefs from trade event (LLM-based inference)"""
        gv2_logger.debug(f"[BG-DEBUG] _update_beliefs_from_trade called for agent {event.agent_id} at price {event.price}")
        if not event.agent_id or event.price is None:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_trade_price = event.price
        agent_node.total_trades += 1
        agent_node.total_volume += event.quantity or 1
        agent_node.last_activity = event.timestamp

        # LLM-based probabilistic belief inference
        self._infer_probabilistic_beliefs_from_event(event)

        # Update asset state
        self.asset_node.last_trade_price = event.price
        self.asset_node.volume_traded += event.quantity or 1

    def _update_beliefs_from_cancel(self, event: MarketEvent) -> None:
        """Update beliefs from cancel event (LLM-based inference)"""
        if not event.agent_id:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_activity = event.timestamp

        # LLM-based probabilistic belief inference
        self._infer_probabilistic_beliefs_from_event(event)

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

        # Update timestamp and confidence
        strategy_edge.confidence = min(self.max_confidence, strategy_edge.confidence + 0.1)
        strategy_edge.timestamp = self.current_time
        strategy_edge.evidence_count += 1

        gv2_logger.debug(f"[BG-STRAT-END] Agent {agent_id} strategy updated (confidence={strategy_edge.confidence:.2f})")
    
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
                    observed_behavior = f"bidding_at_{node.last_bid_price}"
                elif node.last_ask_price:
                    observed_behavior = f"asking_at_{node.last_ask_price}"
                elif node.last_trade_price:
                    if node.total_trades >= 3:
                        observed_behavior = f"frequent_trading_at_{node.last_trade_price}"
                    else:
                        observed_behavior = f"trading_at_{node.last_trade_price}"
                
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
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Market_Direction"] = {
                                "actual_behavior_I_observe": f"observed_trading_activity",
                                f"{node_id.lower()}_self_belief": {
                                    "direction_distribution": edge.value["direction_distribution"]
                                }
                            }
                        
                        elif edge.belief_type == "desperation_level" and "desperation_distribution" in edge.value:
                            competitor_beliefs[f"{node_id}_Beliefs"][f"{node_id}_Desperation_Level"] = {
                                "actual_behavior_I_observe": f"observed_order_pattern",
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
            'probabilistic_beliefs': {
                agent_id: beliefs
                for agent_id, beliefs in self.probabilistic_beliefs.items()
            },
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

        # Reconstruct probabilistic beliefs
        if 'probabilistic_beliefs' in data:
            self.probabilistic_beliefs = data['probabilistic_beliefs']
        
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

        # Extract from edges (works for all graph types)
        if 'strategy' in beliefs and beliefs['strategy']:
            strategy_edge = beliefs['strategy'][0]
            strategy_type = strategy_edge.get('value', strategy_type)

        # For valuation, also try direct edge value
        if 'valuation' in beliefs and beliefs['valuation']:
            val_edge = beliefs['valuation'][0]
            val_value = val_edge.get('value')
            if val_value is not None and not isinstance(val_value, dict):
                valuation_estimate = val_value

        # Add simplified summary
        beliefs['valuation_estimate'] = valuation_estimate
        beliefs['strategy_type'] = strategy_type

        return beliefs


    def get_beliefs(self, agent_id: str) -> str:
        """Get beliefs about an agent (helper method)"""
        beliefs = self.get_agent_beliefs(agent_id)
        return str(beliefs)

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

    @property
    def agents(self):
        """Return list of agent IDs (excluding asset node)"""
        return [node_id for node_id in self.nodes.keys() if node_id != self.asset_id]

    def _add_perfect_beliefs(self, agent_id: str) -> None:
        """
        Add ONLY the original 2 belief types with PERFECT VALUES instead of inferred ones.

        ORIGINAL STRUCTURE MAINTAINED - only strategy and valuation beliefs exist.
        We just cheat by getting perfect values instead of inferring them from behavior.
        """
        trader = self.traders_dict.get(agent_id)
        if not trader:
            return
            
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
                    'strategy': strategy,
                    'valuation_estimate': valuation_estimate,
                    'confidence': valuation_confidence,
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

        # Extract from edges (works for all graph types)
        if 'strategy' in beliefs and beliefs['strategy']:
            strategy_edge = beliefs['strategy'][0]
            strategy_type = strategy_edge.get('value', strategy_type)

        # For valuation, also try direct edge value
        if 'valuation' in beliefs and beliefs['valuation']:
            val_edge = beliefs['valuation'][0]
            val_value = val_edge.get('value')
            if val_value is not None and not isinstance(val_value, dict):
                valuation_estimate = val_value

        # Add simplified summary
        beliefs['valuation_estimate'] = valuation_estimate
        beliefs['strategy_type'] = strategy_type

        return beliefs


    def get_beliefs(self, agent_id: str) -> str:
        """Get beliefs about an agent (helper method)"""
        beliefs = self.get_agent_beliefs(agent_id)
        return str(beliefs)

    def get_market_summary(self) -> Dict[str, Any]:
        """Get a summary of the current market state"""
        return {
            'asset_state': self.asset_node.to_dict(),
            'active_agents': len([n for n in self.nodes.values() if isinstance(n, AgentNode)]),
            'total_beliefs': len(self.edges),
            'recent_events': len(self.event_history),
            'current_time': self.current_time
        }


class GraphVar3:
    """
    Graph Variant 3: LLM-Inferred Belief Traits

    This variant stores beliefs as LLM-inferred trait sets for each agent.
    Instead of numeric scores or probability distributions, it maintains the same
    trait structure that traders use for themselves (aggressiveness, patience, etc.)
    but as beliefs about other traders, inferred by LLM from observed behavior.

    Key idea: "I believe Agent X has aggressiveness=0.8, patience=0.3, risk_tolerance=0.7"
    (inferred by LLM from market observations, not hardcoded rules)
    """

    def __init__(self, asset_id: str = "DEFAULT_ASSET", model=None, logger=None):
        self.graph_id = str(uuid.uuid4())
        self.asset_id = asset_id
        self.nodes: Dict[str, AgentNode | AssetNode] = {}
        self.edges: Dict[str, BeliefEdge] = {}
        self.event_history: List[MarketEvent] = []
        self.current_time = 0.0

        # LLM model for trait inference
        self.model = model
        self.logger = logger or gv3_logger

        # Store belief traits for each agent (flexible dict to support emergent traits)
        self.belief_traits: Dict[str, Dict[str, float]] = {}

        # Initialize the asset node
        self.asset_node = AssetNode(asset_id=asset_id)
        self.nodes[asset_id] = self.asset_node

        # Belief update parameters
        self.valuation_decay_rate = 0.95
        self.confidence_boost = 0.1
        self.max_confidence = 0.95

    def add_agent(self, agent_id: str) -> None:
        """Add a new agent with default belief traits"""
        if agent_id not in self.nodes:
            agent_node = AgentNode(agent_id=agent_id)
            self.nodes[agent_id] = agent_node

            # Initialize with minimal default traits (LLM will generate more)
            self.belief_traits[agent_id] = {
                'confidence': 0.1,
                'reasoning': 'No observations yet'
            }

    @property
    def agents(self):
        """Return list of agent IDs (excluding asset node)"""
        return [node_id for node_id in self.nodes.keys() if node_id != self.asset_id]

    def update_beliefs(self, event: MarketEvent) -> None:
        """Update belief attributes based on market event"""
        self.current_time = event.timestamp
        self.event_history.append(event)

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

        self._update_asset_state()
        self._decay_old_beliefs()

    def _update_beliefs_from_bid(self, event: MarketEvent) -> None:
        """Update belief traits from bid event (LLM-based inference)"""
        if not event.agent_id or event.price is None:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_bid_price = event.price
        agent_node.last_activity = event.timestamp

        # LLM-based trait inference
        self._infer_belief_traits_from_event(event)

    def _update_beliefs_from_ask(self, event: MarketEvent) -> None:
        """Update belief traits from ask event (LLM-based inference)"""
        if not event.agent_id or event.price is None:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_ask_price = event.price
        agent_node.last_activity = event.timestamp

        # LLM-based trait inference
        self._infer_belief_traits_from_event(event)

    def _update_beliefs_from_trade(self, event: MarketEvent) -> None:
        """Update belief traits from trade event (LLM-based inference)"""
        if not event.agent_id or event.price is None:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_trade_price = event.price
        agent_node.total_trades += 1
        agent_node.total_volume += event.quantity or 1
        agent_node.last_activity = event.timestamp

        # Update asset state
        self.asset_node.last_trade_price = event.price
        self.asset_node.volume_traded += event.quantity or 1

        # LLM-based trait inference
        self._infer_belief_traits_from_event(event)

    def _update_beliefs_from_cancel(self, event: MarketEvent) -> None:
        """Update belief traits from cancel event (LLM-based inference)"""
        if not event.agent_id:
            return

        agent_node = self.nodes[event.agent_id]
        agent_node.last_activity = event.timestamp

        # LLM-based trait inference
        self._infer_belief_traits_from_event(event)

    def _update_asset_state(self) -> None:
        """Update the asset node state"""
        if (self.asset_node.current_best_bid is not None and
            self.asset_node.current_best_ask is not None):
            self.asset_node.spread_width = self.asset_node.current_best_ask - self.asset_node.current_best_bid

    def _decay_old_beliefs(self) -> None:
        """Decay confidence in old beliefs"""
        current_time = self.current_time
        for edge in self.edges.values():
            time_diff = current_time - edge.timestamp
            if time_diff > 100:
                decay_factor = self.valuation_decay_rate ** (time_diff / 100)
                edge.confidence *= decay_factor

        # Decay belief trait confidence
        for traits in self.belief_traits.values():
            time_since_update = current_time - self.current_time  # TODO: track per-agent
            if time_since_update > 100:
                decay_factor = self.valuation_decay_rate ** (time_since_update / 100)
                traits.confidence *= decay_factor

    def _build_agent_history(self, agent_id: str) -> Dict[str, Any]:
        """Build trading history for an agent"""
        if agent_id not in self.nodes:
            return {}

        agent_node = self.nodes[agent_id]

        # Extract recent events for this agent
        agent_events = [e for e in self.event_history[-20:] if e.agent_id == agent_id]

        return {
            'total_trades': agent_node.total_trades,
            'total_volume': agent_node.total_volume,
            'last_bid_price': agent_node.last_bid_price,
            'last_ask_price': agent_node.last_ask_price,
            'last_trade_price': agent_node.last_trade_price,
            'last_activity': agent_node.last_activity,
            'recent_events': [
                {
                    'event_type': e.event_type.value,
                    'price': e.price,
                    'quantity': e.quantity,
                    'timestamp': e.timestamp
                } for e in agent_events
            ]
        }

    def _build_market_state(self) -> Dict[str, Any]:
        """Build current market state summary"""
        return {
            'current_best_bid': self.asset_node.current_best_bid,
            'current_best_ask': self.asset_node.current_best_ask,
            'last_trade_price': self.asset_node.last_trade_price,
            'spread_width': self.asset_node.spread_width,
            'volume_traded': self.asset_node.volume_traded,
            'current_time': self.current_time,
            'total_agents': len([n for n in self.nodes.values() if isinstance(n, AgentNode)])
        }

    def _infer_belief_traits_from_event(self, event: MarketEvent) -> None:
        """Infer belief traits using LLM based on observed event"""
        if not self.model or not event.agent_id:
            return

        # Build context
        agent_history = self._build_agent_history(event.agent_id)
        market_state = self._build_market_state()
        current_beliefs = self.belief_traits[event.agent_id]

        # Import here to avoid circular import
        from .unified_prompts import AdaptiveAttributePrompts, PromptParser

        # Build prompt
        prompt = AdaptiveAttributePrompts.infer_belief_traits_prompt(
            event.agent_id,
            event,
            agent_history,
            market_state,
            current_beliefs
        )

        self.logger.info("=== BELIEF TRAIT INFERENCE PROMPT ===")
        self.logger.info("="*80)
        self.logger.info(prompt)
        self.logger.info("="*80)

        # Call LLM
        response = self.model.generate_content(
            prompt,
            generation_config=self.model._generation_config
        )

        self.logger.info("=== BELIEF TRAIT INFERENCE RESPONSE ===")
        self.logger.info(response.text.strip())
        self.logger.info("="*80)

        # Parse and update
        new_traits = PromptParser.parse_attribute_design(response.text)
        old_traits = self.belief_traits[event.agent_id].copy()

        self.update_belief_traits(event.agent_id, new_traits)

        self.logger.info(f"=== BELIEF TRAITS UPDATED FOR {event.agent_id} ===")
        all_keys = set(old_traits.keys()) | set(new_traits.keys())
        for key in sorted(all_keys):
            if key in ['reasoning']:
                continue
            old_val = old_traits.get(key)
            new_val = new_traits.get(key)
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                change = f" ({new_val - old_val:+.2f})"
                self.logger.info(f"{key.replace('_', ' ').title()}: {old_val:.2f} -> {new_val:.2f}{change}")
        self.logger.info("="*80)

    def query_action(self, agent_id: str, current_market_state: Dict[str, Any]) -> Dict[str, Any]:
        """Query the belief graph for decision-making"""
        # Update asset state
        if 'best_bid' in current_market_state:
            self.asset_node.current_best_bid = current_market_state['best_bid']
        if 'best_ask' in current_market_state:
            self.asset_node.current_best_ask = current_market_state['best_ask']
        if 'last_trade' in current_market_state:
            self.asset_node.last_trade_price = current_market_state['last_trade']

        # Build response with belief traits
        belief_data = {
            'graph_id': self.graph_id,
            'current_time': self.current_time,
            'asset_state': {
                'asset_id': self.asset_node.asset_id,
                'current_best_bid': self.asset_node.current_best_bid,
                'current_best_ask': self.asset_node.current_best_ask,
                'last_trade_price': self.asset_node.last_trade_price,
                'spread_width': self.asset_node.spread_width
            },
            'belief_traits': {}
        }

        # Add belief traits for all agents except self
        for node_id in self.agents:
            if node_id != agent_id:
                belief_data['belief_traits'][node_id] = self.belief_traits[node_id]

        return belief_data

    def get_agent_beliefs(self, agent_id: str) -> Dict[str, Any]:
        """Get all beliefs about a specific agent"""
        beliefs = {
            'attributes': self.belief_traits.get(agent_id, {'confidence': 0.1}),
            'traditional': [],
            'valuation_estimate': None,
            'strategy_type': None
        }

        # Get traditional beliefs
        for edge in self.edges.values():
            if edge.target_node == agent_id:
                beliefs['traditional'].append(edge.to_dict())
                if edge.belief_type == "strategy":
                    beliefs['strategy_type'] = edge.value
                elif edge.belief_type == "valuation":
                    beliefs['valuation_estimate'] = edge.value

        return beliefs

    def get_beliefs(self, agent_id: str) -> str:
        """Get beliefs about an agent (helper method)"""
        if agent_id in self.belief_traits:
            traits = self.belief_traits[agent_id]
            trait_strs = [f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                         for k, v in traits.items() if k != 'reasoning']
            return ", ".join(trait_strs) if trait_strs else str(traits)
        return str(self.get_agent_beliefs(agent_id))

    def get_market_summary(self) -> Dict[str, Any]:
        """Get a summary of the current market state"""
        return {
            'asset_state': self.asset_node.to_dict(),
            'active_agents': len([n for n in self.nodes.values() if isinstance(n, AgentNode)]),
            'total_beliefs': len(self.edges),
            'total_belief_trait_sets': len(self.belief_traits),
            'recent_events': len(self.event_history),
            'current_time': self.current_time
        }

    def to_json(self) -> str:
        """Serialize the belief graph to JSON"""
        graph_data = {
            'graph_id': self.graph_id,
            'asset_id': self.asset_id,
            'current_time': self.current_time,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            'belief_traits': {
                agent_id: traits
                for agent_id, traits in self.belief_traits.items()
            },
            'event_history': [event.to_dict() for event in self.event_history[-50:]]
        }
        return json.dumps(graph_data, indent=2)

    def update_belief_traits(self, agent_id: str, traits_dict: Dict[str, float]) -> None:
        """Update belief traits for an agent (supports any trait names from LLM)"""
        if agent_id not in self.belief_traits:
            self.belief_traits[agent_id] = {}

        # Update all traits from dict (allows emergent trait names)
        for key, value in traits_dict.items():
            if key in ['reasoning']:  # Store reasoning as-is
                self.belief_traits[agent_id][key] = value
            elif isinstance(value, (int, float)):  # Clamp numeric traits to [0, 1]
                self.belief_traits[agent_id][key] = max(0.0, min(1.0, value))

        # Synchronize edges with belief traits
        import uuid
        confidence = traits_dict.get('confidence', 0.5)

        for trait_name, trait_value in self.belief_traits[agent_id].items():
            if trait_name in ['reasoning']:
                continue

            # Find and update or create edge for this trait
            edge_found = False
            for edge in self.edges.values():
                if edge.target_node == agent_id and edge.belief_type == trait_name:
                    edge.value = trait_value
                    edge.confidence = confidence
                    edge.timestamp = self.current_time
                    edge_found = True
                    break

            # Create new edge if not found
            if not edge_found:
                new_edge = BeliefEdge(
                    edge_id=str(uuid.uuid4()),
                    source_node=self.asset_id,
                    target_node=agent_id,
                    belief_type=trait_name,
                    confidence=confidence,
                    value=trait_value,
                    timestamp=self.current_time,
                    evidence_count=1
                )
                self.edges[new_edge.edge_id] = new_edge

        self.logger.info(f"[GV3] Updated belief traits for {agent_id}: {self.belief_traits[agent_id]}")
