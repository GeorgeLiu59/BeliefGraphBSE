# Explicit Belief Graph State Management for LLM Market Agents

This implementation provides the data structure for the proposed belief graph system that enables LLM-based trading agents to maintain explicit, continuously-updated beliefs about other agents and market state in multi-agent market simulations.

## Overview

The belief graph addresses the research question: **Do LLM-based trading agents achieve higher profitability and more robust multi-agent coordination when they manage an explicit, continuously-updated belief graph versus relying on unstructured, ever-growing transcripts?**

## Key Features

### 1. **Structured Belief Representation**
- **Nodes**: Represent agents and the traded asset
- **Edges**: Represent beliefs about other agents' valuations, strategies, and intentions
- **Confidence Levels**: Probabilistic belief tracking with confidence scores (0.0 to 1.0)
- **Evidence Tracking**: Count of supporting evidence for each belief

### 2. **Continuous Belief Updates**
- **Market Event Processing**: Automatically updates beliefs based on bids, asks, trades, and cancellations
- **Probabilistic Updates**: Weighted averaging of new evidence with existing beliefs
- **Confidence Decay**: Old beliefs gradually lose confidence over time
- **Strategy Classification**: Automatic classification of agent strategies (aggressive, passive, neutral)

### 3. **LLM-Friendly Interface**
- **JSON Serialization**: Complete graph state can be serialized for LLM consumption
- **Strategic Insights**: Pre-computed analysis of competitors, opportunities, and risks
- **Focused Queries**: Only relevant information is provided to the LLM
- **Compact Representation**: Avoids the "ever-growing transcript" problem

## Architecture

### Core Classes

#### `BeliefGraph`
The main class that manages the entire belief graph state.

#### `AgentNode`
Represents an agent with tracked attributes:
- Last bid/ask/trade prices
- Trading volume and frequency
- Inferred strategy type
- Aggressiveness score
- Valuation estimates

#### `AssetNode`
Represents the traded asset with market state:
- Current best bid/ask
- Spread width
- Volume traded
- Price volatility

#### `BeliefEdge`
Represents beliefs between nodes:
- Belief type (valuation, strategy, intention)
- Confidence level
- Evidence count
- Timestamp

#### `MarketEvent`
Represents market events that update beliefs:
- Event type (bid, ask, trade, cancel)
- Agent ID, price, quantity
- Timestamp

## Usage Example

```python
from belief_graph import BeliefGraph, MarketEvent, EventType

# Create belief graph
graph = BeliefGraph(asset_id="BTC_USD")

# Add agents
graph.add_agent("Alice")
graph.add_agent("Bob")

# Process market events
event = MarketEvent(
    event_id="event_001",
    event_type=EventType.BID,
    timestamp=1.0,
    agent_id="Alice",
    price=100.0,
    quantity=1
)
graph.update_beliefs(event)

# Query for LLM decision-making
market_state = {'best_bid': 101.0, 'best_ask': 103.0}
belief_data = graph.query_action("Bob", market_state)

# Serialize for LLM consumption
json_data = graph.to_json()
```

## Integration with BSE

The belief graph is designed to integrate seamlessly with the Bristol Stock Exchange (BSE) system:

1. **Event Integration**: BSE market events can be converted to `MarketEvent` objects
2. **Agent Integration**: BSE traders can maintain their own belief graph instance
3. **Decision Making**: The `query_action()` method provides structured data for LLM-based decision making

## Research Contributions

### 1. **Addresses Existing Gaps**
- **Structured Beliefs**: Unlike free-form text beliefs in existing work
- **Continuous Updates**: Unlike static belief graphs
- **Market-Specific**: Unlike general-purpose knowledge graphs
- **Hidden Information**: Unlike perfect-information games

### 2. **LLM State Management**
- **Scalable**: Beliefs scale with number of agents, not time
- **Focused**: Only relevant beliefs are provided to LLM
- **Queryable**: Structured format enables symbolic reasoning
- **Compact**: Avoids context window limitations

### 3. **Strategic Decision Making**
- **Competitor Analysis**: Tracks other agents' strategies and valuations
- **Opportunity Detection**: Identifies arbitrage and trading opportunities
- **Risk Assessment**: Monitors market volatility and risk factors
- **Utility Inference**: Estimates other agents' payoff functions

## Testing

Run the test suite to see the belief graph in action:

```bash
python test_belief_graph.py
```

This demonstrates:
- Basic belief graph functionality
- Market event processing
- LLM query interface
- JSON serialization
- Extended trading scenarios

## Future Work

### 1. **BSE Integration**
- Create BSE trader subclass that uses belief graph
- Integrate with BSE event system
- Add belief graph to market session management

### 2. **LLM Integration**
- Implement LLM-based decision making using belief graph data
- Add natural language querying of belief graph
- Create prompts that leverage structured beliefs

### 3. **Advanced Features**
- Multi-level belief hierarchies
- Temporal belief modeling
- Uncertainty quantification
- Belief conflict resolution

### 4. **Performance Optimization**
- Efficient belief update algorithms
- Belief pruning and compression
- Distributed belief sharing

## Files

- `belief_graph.py`: Main implementation
- `test_belief_graph.py`: Test suite and examples
- `BELIEF_GRAPH_README.md`: This documentation

## Research Context

This implementation directly addresses the gaps identified in the research proposal:

1. **Guandan ToM-LLMs**: Provides structured beliefs instead of free-form text
2. **REFLEX**: Enables online belief updates instead of offline consistency checking
3. **Hypothetical Minds**: Creates discrete, probabilistic beliefs instead of unstructured hypotheses
4. **Game-Theoretic Workflows**: Handles hidden information instead of assuming common knowledge
5. **COKE**: Provides real-time updates instead of static knowledge graphs

The belief graph enables LLM-based trading agents to maintain sophisticated models of other agents' beliefs and intentions, leading to more effective strategic decision-making in multi-agent market environments.
