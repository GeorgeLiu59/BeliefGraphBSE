Fork of https://github.com/davecliff/BristolStockExchange with Beliefs Graphs added to agents.

## New Contributions

### Explicit Belief Graph State Management for LLM Market Agents

A novel belief graph implementation that enables LLM-based trading agents to maintain explicit, continuously-updated beliefs about other agents and market state. The belief graph tracks agent behaviors, strategies, and market dynamics through structured nodes and probabilistic belief edges.

**Key Features:**
- Structured belief representation with confidence scoring
- Continuous belief updates based on market events
- LLM-friendly JSON serialization and query interface
- Strategic insights generation for decision-making

**Files:**
- `belief_graph.py`: Main implementation
- `BELIEF_GRAPH_README.md`: Detailed documentation and usage examples

### Performance Analysis Tool

The `analyze_performance.py` script provides comprehensive analysis of BSE simulation results, enabling detailed evaluation of trading agent performance.

**Features:**
- Performance metrics (balance, profit, risk-adjusted returns)
- Risk analysis (Sharpe ratio, volatility, max drawdown)
- Visualization (balance evolution, comparative charts)
- Automated parsing of BSE output files

**Usage:**
```bash
python analyze_performance.py                    # Analyze default files
python analyze_performance.py file1.csv file2.csv  # Analyze specific files
```

**Output:**
- Console performance summary and risk metrics
- Performance visualization (`agent_performance_analysis.png`)
- Transaction statistics and market analysis

This tool enables researchers to systematically evaluate different trading strategies and understand market dynamics in the BSE simulation environment.


 
