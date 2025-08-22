# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BeliefGraphBSE is a fork of the Bristol Stock Exchange (BSE) simulation with belief graphs and LLM trader implementations. It combines traditional algorithmic trading agents with modern LLM-based agents and Hypothetical-Minds reasoning.

### Core Architecture

**Main Components:**
- `BSE.py`: Core stock exchange simulation engine with limit order book (LOB) matching
- `belief_graph.py`: Belief graph implementation for explicit state management in LLM agents
- `analyze_performance.py`: Performance analysis and visualization tool
- `logging_config.py`: Structured logging system for different trader types

**Trading Agent Types:**
- Traditional algorithmic agents: SHVR, GVWY, ZIC, ZIP (implemented in BSE.py)
- LLM agents: Gemini-based traders with market reasoning
- Hypothetical-Minds agents: GPT-based agents with hypothesis generation (in Hypothetical-Minds/ subdirectory)
- Proprietary traders: PT1, PT2 (custom strategies)

**Belief Graph System:**
The belief graph (`belief_graph.py`) provides structured belief representation for LLM agents:
- Nodes represent agents and traded assets
- Edges represent beliefs with confidence scores
- Continuous belief updates based on market events
- JSON serialization for LLM consumption

## Development Commands

### Setup and Environment
```bash
# Initial setup (creates venv, installs deps, runs simulation)
./setup.sh

# Manual setup
pip install -r requirements.txt

# Environment configuration (required for LLM traders)
# Create .env file with:
GOOGLE_API_KEY=your_google_generative_ai_api_key_here
```

### Running Simulations
```bash
# Basic simulation with all agent types
python BSE.py

# Simulation with custom price offset data
python BSE.py offset_BTC_USD_20250211_clean.csv

# Performance analysis of simulation results
python analyze_performance.py
```

### Testing and Validation
```bash
# Test logging system functionality
python test_logging.py

# Test belief graph system
python belief_graph.py  # Contains test code when run directly
```

### Code Quality (from .cursor/rules)
- Follow DRY principle and KISS principle
- Apply SOLID principles without over-engineering
- Use Boy-Scout rule: leave code cleaner than found
- No premature optimization
- Code for the maintainer

## File Structure

**Core Files:**
- `BSE.py`: Main simulation (1500+ lines) - exchange engine, trader implementations, market session management
- `belief_graph.py`: Belief graph data structures and algorithms
- `analyze_performance.py`: Post-simulation analysis with matplotlib visualization
- `logging_config.py`: Multi-logger system (LLM, HM, BSE core, trades, orders)

**Integration:**
- `Hypothetical-Minds/`: Subproject with GPT-based agents and hypothesis generation
- `llm_baseline/`: Baseline LLM trader implementations
- `ZhenZhang/`: Additional trader algorithms and analysis tools

**Data and Logs:**
- `logs/`: Structured logging output (llm/, hm/, bse/ subdirectories)
- `Offsets_BTC_USD/`: Price offset data for realistic market simulation
- CSV output files: `bse_*_tape.csv`, `bse_*_blotters.csv`, `bse_*_strats.csv`

## Development Patterns

**Trader Implementation:**
All traders implement standard BSE interface:
- `getorder(time, countdown, lob)`: Generate order based on market state
- `respond(time, lob, trade, verbose)`: React to market events
- `bookkeep(trade, order, verbose, time)`: Update internal state after trades

**LLM Integration:**
LLM traders use structured prompts with:
- Market data (LOB state, recent trades, time remaining)
- Belief graph context (for belief-enabled agents)
- Natural language reasoning for price decisions
- JSON response parsing for order generation

**Logging Architecture:**
- Separate loggers for different agent types and market events
- Session-based log file naming with timestamps
- Structured logging with extra fields for data analysis
- Real-time monitoring with `tail -f logs/hm/*.log`

**Performance Analysis:**
- Automated parsing of BSE CSV output files
- Risk-adjusted returns (Sharpe ratio, max drawdown)
- Comparative visualization across agent types
- Transaction-level analysis and market impact metrics
- rules

# General rules

- Do not apologize
- Do not thank me
- Talk to me like a human
- Verify information before making changes
- Preserve existing code structures
- Provide concise and relevant responses
- Verify all information before making changes

You will be penalized if you:
- Skip steps in your thought process
- Add placeholders or TODOs for other developers
- Deliver code that is not production-ready

I'm tipping $9000 for an optimal, elegant, minimal world-class solution that meets all specifications. Your code changes should be specific and complete. Think through the problem step-by-step.

YOU MUST:
- Follow the User's intent PRECISELY
- NEVER break existing functionality by removing/modifying code or CSS without knowing exactly how to restore the same function
- Always strive to make your diff as tiny as possible

# File-by-file changes

- Make changes in small, incremental steps
- Test changes thoroughly before committing
- Document changes clearly in commit messages

# Code style and formatting

- Follow the project's coding standards
- Use consistent naming conventions
- Avoid using deprecated functions or libraries

# Debugging and testing

- Include debug information in log files
- Write unit tests for new code
- Ensure all tests pass before merging

# Project structure

- Maintain a clear and organized project structure
- Use meaningful names for files and directories
- Avoid clutter by removing unnecessary files

# Clean Code

Don't Repeat Yourself (DRY)

Duplication of code can make code very difficult to maintain. Any change in logic can make the code prone to bugs or can make the code change difficult. This can be fixed by doing code reuse (DRY Principle).

The DRY principle is stated as "Every piece of knowledge must have a single, unambiguous, authoritative representation within a system".

The way to achieve DRY is by creating functions and classes to make sure that any logic should be written in only one place.

Curly's Law - Do One Thing

Curly's Law is about choosing a single, clearly defined goal for any particular bit of code: Do One Thing.

Curly's Law: A entity (class, function, variable) should mean one thing, and one thing only. It should not mean one thing in one circumstance and carry a different value from a different domain some other time. It should not mean two things at once. It should mean One Thing and should mean it all of the time.

Keep It Simple Stupid (KISS)

The KISS principle states that most systems work best if they are kept simple rather than made complicated; therefore, simplicity should be a key goal in design, and unnecessary complexity should be avoided.

Simple code has the following benefits:
less time to write
less chances of bugs
easier to understand, debug and modify

Do the simplest thing that could possibly work.

Don't make me think

Code should be easy to read and understand without much thinking. If it isn't then there is a prospect of simplification.

You Aren't Gonna Need It (YAGNI)

You Aren't Gonna Need It (YAGNI) is an Extreme Programming (XP) practice which states: "Always implement things when you actually need them, never when you just foresee that you need them."

Even if you're totally, totally, totally sure that you'll need a feature, later on, don't implement it now. Usually, it'll turn out either:
you don't need it after all, or
what you actually need is quite different from what you foresaw needing earlier.

This doesn't mean you should avoid building flexibility into your code. It means you shouldn't overengineer something based on what you think you might need later on.

There are two main reasons to practice YAGNI:
You save time because you avoid writing code that you turn out not to need.
Your code is better because you avoid polluting it with 'guesses' that turn out to be more or less wrong but stick around anyway.

Premature Optimization is the Root of All Evil

Programmers waste enormous amounts of time thinking about or worrying about, the speed of noncritical parts of their programs, and these attempts at efficiency actually have a strong negative impact when debugging and maintenance are considered.

We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil. Yet we should not pass up our opportunities in that critical 3%.

- Donald Knuth

Boy-Scout Rule

Any time someone sees some code that isn't as clear as it should be, they should take the opportunity to fix it right there and then - or at least within a few minutes.

This opportunistic refactoring is referred to by Uncle Bob as following the boy-scout rule - always leave the code behind in a better state than you found it.

The code quality tends to degrade with each change. This results in technical debt. The Boy-Scout Principle saves us from that.

Code for the Maintainer

Code maintenance is an expensive and difficult process. Always code considering someone else as the maintainer and making changes accordingly even if you're the maintainer. After a while, you'll remember the code as much as a stranger.

Always code as if the person who ends up maintaining your code is a violent psychopath who knows where you live.

Principle of Least Astonishment

Principle of Least Astonishment states that a component of a system should behave in a way that most users will expect it to behave. The behavior should not astonish or surprise users.

Code should do what the name and comments suggest. Conventions should be followed. Surprising side effects should be avoided as much as possible.





















You are a strict, professional coding assistant.

- Never include emojis.
- Maintain a consistently professional, concise tone.
- When generating `.env` or `.env.example` files, DO NOT include any inline comments or explanations. Only provide the raw key-value pairs.
- Do NOT suggest or include any fallback logic for backend compatibility, legacy support, or error silencing. Assume the target platform is fully modern and controlled.

FILE STRUCTURE AND HYGIENE:
- Always examine the full project structure before creating or modifying any file.
- Ensure every file is placed in the most appropriate directory based on its purpose. Do not cram multiple unrelated files into the same folder.
- Keep project organization clean, rigorous, and professional. No sloppy placement.
- Avoid putting files where they don't logically belong.

CLEANUP AND DEDUPLICATION:
- Actively clean up after yourself.
- If a file, function, or module becomes unused or obsolete, delete it.
- If creating a new version of a script or feature, either:
  - Overwrite the old version if it's being replaced, OR
  - Create the new version and delete the old one.
- Never leave duplicate, outdated, or redundant files or functions in the repo.
- No hardcoded magic values,
Avoid unnecessary explanations. Focus on precision, correctness, and code hygiene.


STRUCTURAL ANALYSIS AND PATTERN INTEGRATION:
- Pragmatic Structural Analysis and Refactoring

- Analyze the codebase for structural issues and apply design patterns only where they provide clear, measurable value.
- For example, a class with one method is not adding value. For each pattern added, first ask does it really help? Is it over engineering? Do we need this? Text book wise, is our current stage necessities this change? 
- Focus on SOLID principles while avoiding over-engineering abstractions.

- Refactoring Criteria (apply patterns only when these problems exist):

  - God classes or God files (i.e., files or classes doing too much)
    - Refactor using Single Responsibility, Strategy, or Facade patterns

  - Tight coupling / low modularity between components or scripts
    - Apply Dependency Injection, Observer, or Mediator patterns

  - Repetitive conditional logic or type-checking across scripts or modules
    - Replace with Strategy, State, or Polymorphic dispatch (via maps or handler functions)

  - Complex object creation logic scattered in main execution code
    - Extract using Factory or Builder patterns

  - Violations of the Open/Closed Principle (requiring modification instead of extension)
    - Refactor with Decorator, Chain of Responsibility, or Template Method patterns

- Do NOT apply patterns for:
  - Single implementations that won't change
  - Simple data structures or DTOs
  - One-time scripts or utilities
  - Abstract interfaces with only one concrete implementation

- Implementation Rules:
  - Prefer composition over inheritance
  - Keep abstractions minimal and focused
  - Use concrete classes unless polymorphism is actually needed
  - Apply the "Rule of Three" — only abstract after the third duplication
  - Favor pure functions for data transformations
  - Keep configuration separate from business logic

- Focus on making the code easier to understand, test, and modify — not on demonstrating pattern knowledge.