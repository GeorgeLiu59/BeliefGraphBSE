"""
Agent Configuration Module - Mirror of HM's agent_config.py

Maps different trading strategies to their implementations, following
the exact same pattern as Hypothetical-Minds agent configuration.
"""

# BSE Trading Agent Configuration
# Mirrors: llm_plan/agent/agent_config.py structure
bse_agent_config = {
    'bristol_stock_exchange__trading': {
        'hm': {
            'gemini': 'hm_trader.bse_trading.bse_hm_gemini.DecentralizedAgent',
            'gpt4': 'hm_trader.bse_trading.bse_hm_gpt4.DecentralizedAgent',
            'gpt35': 'hm_trader.bse_trading.bse_hm_gpt35.DecentralizedAgent',
            # Note: We'll only implement gemini for now but maintain HM structure
        },
        'react': {
            # Note: ReAct strategy could be added later
            'gemini': 'hm_trader.bse_trading.bse_react_gemini.DecentralizedAgent',
        },
        'reflexion': {
            # Note: Reflexion strategy could be added later  
            'gemini': 'hm_trader.bse_trading.bse_reflexion_gemini.DecentralizedAgent',
        },
        'planreact': {
            # Note: PlanReAct strategy could be added later
            'gemini': 'hm_trader.bse_trading.bse_planreact_gemini.DecentralizedAgent',
        },
    }
}

# Model-specific parameters
MODEL_PARAMS = {
    'gemini-2.5-flash-lite': {
        'temperature': 0.7,
        'max_tokens': 32000,  # Increased for full reasoning text
    },
    'gpt-4': {
        'temperature': 0.7,
        'max_tokens': 32000,  # Increased for full reasoning text
    },
    'gpt-3.5-turbo': {
        'temperature': 0.7,
        'max_tokens': 32000,  # Increased for full reasoning text
    }
}