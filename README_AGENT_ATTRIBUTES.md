# AI Agent Self-Designed Attributes System

## Overview

This system allows **LLM agents to design and adapt their own trading attributes** that influence how they interact with the belief graph and make trading decisions. Agents can:

1. **Design their initial trading personality** at the start of a simulation
2. **Adapt their attributes** based on performance and market conditions
3. **Use attributes to influence belief graph interactions** and trading decisions

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Attribute System                   │
├─────────────────────────────────────────────────────────────┤
│  AttributeManager  │  AttributeDesigner  │  AttributeAdapter │
│  • Manages agent  │  • Creates initial  │  • Adapts        │
│    attributes     │    attribute sets   │    attributes    │
│  • Handles       │  • 6 design         │  • Based on      │
│    persistence   │    strategies       │    performance   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  LLM Integration Layer                     │
├─────────────────────────────────────────────────────────────┤
│  AttributePromptTemplates  │  AttributePromptParser      │
│  • Initial design prompts  │  • Parses LLM responses     │
│  • Adaptation prompts      │  • Extracts strategies      │
│  • Trading prompts         │  • Handles fallbacks        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  TraderAdaptive Class                     │
├─────────────────────────────────────────────────────────────┤
│  • Integrates with belief graph                          │
│  • Uses attributes for trading decisions                 │
│  • Automatically adapts based on performance             │
│  • Records adaptation history                            │
└─────────────────────────────────────────────────────────────┘
```

## Attribute Types

### 1. **Aggressiveness** (0.0 - 1.0)
- **Low (0.0-0.3)**: Conservative, waits for optimal conditions
- **Medium (0.4-0.6)**: Balanced approach to trading
- **High (0.7-1.0)**: Aggressive, actively pursues opportunities

### 2. **Risk Tolerance** (0.0 - 1.0)
- **Low (0.0-0.3)**: Prefers safe, predictable trades
- **Medium (0.4-0.6)**: Accepts moderate risk
- **High (0.7-1.0)**: Accepts high uncertainty and volatility

### 3. **Patience** (0.0 - 1.0)
- **Low (0.0-0.3)**: Trades frequently, impatient
- **Medium (0.4-0.6)**: Moderate waiting periods
- **High (0.7-1.0)**: Waits for very good opportunities

### 4. **Adaptability** (0.0 - 1.0)
- **Low (0.0-0.3)**: Sticks to initial strategy
- **Medium (0.4-0.6)**: Moderately adaptable
- **High (0.7-1.0)**: Quickly changes strategies

### 5. **Momentum Following** (0.0 - 1.0)
- **Low (0.0-0.3)**: Ignores market trends
- **Medium (0.4-0.6)**: Partially follows trends
- **High (0.7-1.0)**: Strongly follows momentum

### 6. **Mean Reversion** (0.0 - 1.0)
- **Low (0.0-0.3)**: Follows trends
- **Medium (0.4-0.6)**: Balanced approach
- **High (0.7-1.0)**: Bets on price reversals

## Design Strategies

### **Conservative Strategy**
- Low aggressiveness, low risk tolerance
- High patience, high mean reversion
- Good for stable markets, low volatility

### **Aggressive Strategy**
- High aggressiveness, high risk tolerance
- Low patience, high momentum following
- Good for volatile markets, trend following

### **Balanced Strategy**
- Moderate values across all attributes
- Good for mixed market conditions
- Default fallback strategy

### **Momentum Strategy**
- High momentum following, low mean reversion
- Medium-high aggressiveness
- Good for trending markets

### **Mean Reversion Strategy**
- High mean reversion, low momentum following
- High patience, medium risk tolerance
- Good for range-bound markets

### **Random Strategy**
- Random values across all attributes
- Used for exploration and testing

## Adaptation System

### **Automatic Adaptation Triggers**
- **Poor Performance**: Losses > $50
- **High Volatility**: Market volatility > 0.8
- **Underperformance**: Relative performance < -0.2

### **Adaptation Strategies**
- **Conservative**: Applied after big losses (>$100)
- **Balanced**: Applied during high volatility
- **Aggressive**: Applied when underperforming significantly

### **Adaptation Process**
1. **Monitor Performance**: Track profit, volatility, relative performance
2. **Trigger Check**: Determine if adaptation is needed
3. **Strategy Selection**: Choose appropriate adaptation strategy
4. **Attribute Blending**: Blend new attributes with current ones based on adaptability
5. **History Recording**: Log all adaptations for analysis

## Usage Examples

### **Basic Usage**

```python
from agent_attributes import AttributeManager
from TraderAdaptive import TraderAdaptive

# Create an attribute manager
manager = AttributeManager("agent_001")
manager.initialize_attributes("momentum")

# Create adaptive trader
trader = TraderAdaptive(
    ttype="ADAPTIVE",
    tid="agent_001", 
    balance=1000.0,
    params={'adaptation_enabled': True},
    time=0.0
)

# Initialize attributes (LLM will design them)
trader.initialize_attributes()

# Check if adaptation is needed
trader.check_and_adapt_attributes()
```

### **Custom Design Strategy**

```python
# Use specific design strategy
manager.initialize_attributes("conservative")

# Or design custom attributes
from agent_attributes import AttributeDesigner
custom_attrs = AttributeDesigner.design_conservative()
```

### **Performance Monitoring**

```python
# Get current attributes
attributes = trader.get_attributes()
print(f"Current aggressiveness: {attributes.aggressiveness:.2f}")

# Get adaptation summary
summary = trader.get_attributes_summary()
print(f"Design strategy: {summary['design_strategy']}")
print(f"Adaptations: {len(summary['adaptation_history'])}")
```

## 📈 Analysis and Visualization

### **LLM Attribute Analyzer**

The `analyze_llm_attributes.py` module provides comprehensive analysis:

```python
from analyze_llm_attributes import LLMAttributeAnalyzer

analyzer = LLMAttributeAnalyzer()

# Add agent data
analyzer.add_agent_data("agent_001", attribute_data)

# Run analysis
report_file = analyzer.generate_report()
viz_files = analyzer.generate_visualizations()

# Print summary
analyzer.print_summary()
```

### **Generated Visualizations**
1. **Attribute Distribution Heatmap**: Shows correlations between attributes
2. **Design Strategy Distribution**: Bar chart of strategy popularity
3. **Adaptation Timeline**: Frequency of adaptations over time
4. **Performance Correlation Scatter**: Attributes vs performance plots

### **Analysis Reports**
- **Attribute Distributions**: Statistical analysis of attribute values
- **Design Strategy Analysis**: Which strategies agents choose and why
- **Adaptation Patterns**: When and how agents adapt
- **Performance Correlations**: Which attributes correlate with success
- **Agent Evolution**: Individual agent adaptation history

## Integration with BSE

### **Adding to BSE.py**

```python
# In the trader factory section
elif robottype == "ADAPTIVE":
    trader = TraderAdaptive(
        ttype=robottype,
        tid=tid,
        balance=balance,
        params=trader_params,
        time=time
    )
```

### **Configuration Options**

```python
trader_params = {
    'api_key': 'your_google_api_key',  # For LLM integration
    'adaptation_enabled': True,         # Enable/disable adaptation
    'adaptation_interval': 10,          # Check adaptation every N trades
    'temperature': 0.3,                 # LLM creativity level
    'model_name': 'gemini-2.0-flash-lite'
}
```

### **Belief Graph Integration**

The `TraderAdaptive` class automatically:
- Updates the belief graph with market events
- Uses belief graph insights for trading decisions
- Influences belief formation based on agent attributes

## Testing

### **Run Basic Tests**

```bash
python3 test_adaptive_trader.py
```

### **Run Analysis Demo**

```bash
python3 analyze_llm_attributes.py
```

### **Test Individual Components**

```python
# Test attribute design
from agent_attributes import AttributeDesigner
attrs = AttributeDesigner.design_aggressive()
print(attrs.to_dict())

# Test prompt templates
from llm_attribute_prompts import AttributePromptTemplates
prompt = AttributePromptTemplates.create_initial_design_prompt(
    market_context, 
    AttributeDesigner.get_design_strategies()
)
```

## File Structure

```
BeliefGraphBSE/
├── agent_attributes.py          # Core attribute system
├── llm_attribute_prompts.py     # LLM prompt templates
├── TraderAdaptive.py           # Adaptive trader implementation
├── test_adaptive_trader.py     # System tests
├── analyze_llm_attributes.py   # Analysis and visualization
└── README_AGENT_ATTRIBUTES.md  # This file
```

## API Keys

To use the LLM integration, you need a Google API key:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

Or pass it in the trader parameters:

```python
trader_params = {'api_key': 'your_api_key'}
```

## Error Handling

### **Common Issues**

1. **No API Key**: Falls back to balanced strategy
2. **LLM Errors**: Uses fallback decision logic
3. **Belief Graph Import Error**: Continues without belief graph
4. **Invalid Attributes**: Validates and corrects automatically

### **Fallback Behavior**

- **Attribute Design**: Defaults to "balanced" strategy
- **Trading Decisions**: Falls back to "WAIT" decision
- **Adaptation**: Continues with current attributes

## Future Enhancements

### **Planned Features**
1. **Multi-Agent Coordination**: Agents learn from each other
2. **Market Regime Detection**: Automatic strategy switching
3. **Advanced Adaptation**: Machine learning-based adaptation
4. **Real-time Analysis**: Live attribute performance monitoring

### **Research Applications**
1. **Agent Behavior Analysis**: Study how LLMs design trading personalities
2. **Adaptation Effectiveness**: Measure which adaptations work best
3. **Market Impact**: Analyze how agent attributes affect market dynamics
4. **Belief Graph Evolution**: Study belief formation patterns

## References

- **Bristol Stock Exchange (BSE)**: Base simulation framework
- **Belief Graph**: State management and belief formation
- **Google Gemini**: LLM for attribute design and adaptation
- **Multi-Agent Systems**: Theory behind adaptive agents

## Contributing

To contribute to this system:

1. **Add New Design Strategies**: Extend `AttributeDesigner` class
2. **Improve Adaptation Logic**: Enhance `AttributeAdapter` class
3. **Add New Analysis Methods**: Extend `LLMAttributeAnalyzer` class
4. **Create New Trader Types**: Extend `TraderAdaptive` class

## License

This system is part of the BeliefGraphBSE project. See the main LICENSE file for details.

---

**The goal is to create truly autonomous trading agents that can design, adapt, and evolve their own trading personalities based on market conditions and performance feedback.**
