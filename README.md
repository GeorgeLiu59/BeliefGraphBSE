Fork of https://github.com/davecliff/BristolStockExchange with Beliefs Graphs added.

## Installation

### Prerequisites
- Python 3.7+

### Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration:**
   Create a `.env` file in the project root with your API key:
   ```bash
   GOOGLE_API_KEY=your_google_generative_ai_api_key_here
   ```
   
   **Note:** You'll need a Google Generative AI API key to use the LLM traders. Get one from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Getting Started

### Running the BSE Simulation

1. **Basic Simulation:**
   ```bash
   python BSE.py
   ```
   This runs a default market session with various trading agents including the new LLM traders.

2. **Custom Parameters:**
   ```bash
   python BSE.py [price_offset_filename]
   ```
   Run with custom price offset data (see `Offsets_BTC_USD/` for examples).

### Analyzing Results

After running a simulation, analyze the results:
```bash
python analyze_performance.py
```
This generates performance metrics and visualizations from the simulation output files.

### Key Files

- `BSE.py`: Main simulation engine
- `BSE_VernonSmith1962_demo.ipynb`: Interactive tutorial
- `analyze_performance.py`: Performance analysis tool
- `belief_graph.py`: Belief graph implementation

## New Contributions

### LLM Trader Implementation

A baseline LLM-based trading agent using Google's Generative AI (Gemini) for market decision-making. The LLM trader integrates with the BSE simulation by receiving real-time market data (limit order book, trade history, time remaining) and generating quote prices through natural language reasoning.

**How it Works:**
- Receives market data from BSE including current LOB state, recent trades, and trader context
- Formats this data into natural language prompts for the LLM
- Uses Google's Gemini model to generate price decisions based on market conditions
- Submits quotes back to the BSE exchange for order matching
- Maintains trading history for context in subsequent decisions

**BSE Integration:**
The LLM trader implements the standard BSE trader interface (`getorder()` and `respond()` methods), allowing it to participate seamlessly alongside other algorithmic traders. It processes the same market events and follows the same trading rules as traditional agents.

**Usage:**
The LLM trader is automatically included in simulation runs when a valid API key is provided in the `.env` file.

### Explicit Belief Graph State Management for LLM Market Agents

A novel belief graph implementation that enables LLM-based trading agents to maintain explicit, continuously-updated beliefs about other agents and market state. The belief graph tracks agent behaviors, strategies, and market dynamics through structured nodes and probabilistic belief edges.

**Key Features:**
- Structured belief representation with confidence scoring
- Continuous belief updates based on market events
- LLM-friendly JSON serialization and query interface
- Strategic insights generation for decision-making

**Files:**
- `belief_graph.py`: Main implementation
- `BELIEF_GRAPH.md`: Detailed documentation and usage examples

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
```

**Output:**
- Console performance summary and risk metrics
- Performance visualization (`agent_performance_analysis.png`)
- Transaction statistics and market analysis

This tool helps us understand the unstructured output from running BSE.


 
