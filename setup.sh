#!/bin/bash

# BeliefGraphBSE Setup Script
# Sets up environment following Hypothetical-Minds README pattern

set -e

echo "Setting up BeliefGraphBSE with Hypothetical-Minds integration..."

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required. Please install Python 3.7+ and try again."
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Setup environment variables (following HM README)
if [ ! -f ".env" ]; then
    echo "️  Creating .env file..."
    cat > .env << EOF
GOOGLE_API_KEY=your_google_generative_ai_api_key_here
EOF
    echo "Please edit .env file with your actual API key"
else
    echo ".env file already exists"
fi

# Test installation
echo "Testing installation..."
python3 -c "
try:
    import numpy, matplotlib, google.generativeai, omegaconf
    from dotenv import load_dotenv
    print('All dependencies imported successfully')
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
"

# Skip integration test to avoid circular imports
echo "️  Skipping integration test (will test during simulation)"

# Setup logging directory
echo "Setting up logging system..."
if [ ! -d "logs" ]; then
    mkdir -p logs
    echo "Created logs directory"
else
    echo "Logs directory already exists"
    # Clear old logs and create the 3 main files
    rm -f logs/*.txt logs/*.log
    touch logs/general_output.txt logs/llm_agents.txt logs/hm_agents.txt
    echo "Cleared old logs and created 3 main log files"
fi

echo ""
echo "Setup complete!"
echo ""
echo "RUNNING BSE SIMULATION WITH ALL AGENTS..."
echo ""

# Run BSE simulation
echo "Starting BSE simulation with:"
echo "   - Traditional agents: SHVR, GVWY, ZIC, ZIP"
echo "   - George's LLM agents: LLM (Gemini)"
echo "   - Hypothetical-Minds agents: HM_GPT4, HM_REACT"
echo "   - Proprietary traders: PT1, PT2"
echo ""

python3 BSE.py >> logs/general_output.txt 2>&1

echo ""
echo "ANALYZING RESULTS..."
echo ""

# Analyze results
python3 analyze_performance.py >> logs/general_output.txt 2>&1

echo ""
echo "SIMULATION COMPLETE!"
echo ""
echo "Results show performance comparison between:"
echo "   - Traditional algorithmic traders"
echo "   - George's Gemini-based LLM traders"
echo "   - Hypothetical-Minds GPT-based agents"
echo ""
echo "Check the generated outputs:"
echo "   - Performance chart: key_performance_metrics.png"
echo "   - Detailed logs: logs/"
echo "     • General simulation output: logs/general_output.txt"
echo "     • LLM trader decisions: logs/llm_agents.txt"
echo "     • HM trader activity (including hypotheses): logs/hm_agents.txt"
echo ""
echo "To view real-time logs during simulation, tail the log files:"
echo "   tail -f logs/hm_agents.txt  # Watch HM trader activity and hypotheses"
echo "   tail -f logs/llm_agents.txt  # Watch LLM trader decisions"
echo "   tail -f logs/general_output.txt  # Watch general simulation output"
echo ""
