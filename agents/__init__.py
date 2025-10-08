#!/usr/bin/env python3
"""
Agents Package
Exports all trading agent classes and factory for easy instantiation
"""

from agents.base_llm_trader import BaseLLMTrader
from agents.trader_llm_baseline import TraderLLM_Baseline
from agents.trader_bg import TraderBG
from agents.trader_pg import TraderPG
from agents.trader_gv1 import TraderGV1
from agents.trader_gv2 import TraderGV2
from agents.trader_gv3 import TraderGV3
from agents.trader_hm import TraderHM

__all__ = [
    'BaseLLMTrader',
    'TraderLLM_Baseline',
    'TraderBG',
    'TraderPG',
    'TraderGV1',
    'TraderGV2',
    'TraderGV3',
    'TraderHM',
    'AgentFactory'
]


class AgentFactory:
    """
    Factory for creating all 25 agent variants

    Agent naming convention:
    - LLM: Base LLM (no belief graph)
    - BG: Basic Graph
    - PG: Perfect Graph
    - GV1: Graph Variance 1 (discrete sets)
    - GV2: Graph Variance 2 (probabilistic)
    - GV3: Graph Variance 3 (LLM-designed attributes)
    - HM: Hypothetical Minds

    Suffixes:
    - CO: With Chain-of-Thought
    - NO: No Chain-of-Thought
    - JSON: JSON belief format
    - NL: Natural Language belief format
    """

    AGENT_CONFIGS = {
        # 1. LLM Baseline
        'LLM': {
            'class': TraderLLM_Baseline,
            'params': {'use_cot': False, 'belief_format': 'json'}
        },

        # 2-5. Basic Graph variants
        'BG_JSON_COT': {
            'class': TraderBG,
            'params': {'use_cot': True, 'belief_format': 'json'}
        },
        'BG_NL_COT': {
            'class': TraderBG,
            'params': {'use_cot': True, 'belief_format': 'nl'}
        },
        'BG_JSON_NOCOT': {
            'class': TraderBG,
            'params': {'use_cot': False, 'belief_format': 'json'}
        },
        'BG_NL_NOCOT': {
            'class': TraderBG,
            'params': {'use_cot': False, 'belief_format': 'nl'}
        },

        # 6-9. Perfect Graph variants
        'PG_JSON_COT': {
            'class': TraderPG,
            'params': {'use_cot': True, 'belief_format': 'json'}
        },
        'PG_NL_COT': {
            'class': TraderPG,
            'params': {'use_cot': True, 'belief_format': 'nl'}
        },
        'PG_JSON_NOCOT': {
            'class': TraderPG,
            'params': {'use_cot': False, 'belief_format': 'json'}
        },
        'PG_NL_NOCOT': {
            'class': TraderPG,
            'params': {'use_cot': False, 'belief_format': 'nl'}
        },

        # 10-13. Graph Variance 1 (Discrete Sets)
        'GV1_JSON_COT': {
            'class': TraderGV1,
            'params': {'use_cot': True, 'belief_format': 'json'}
        },
        'GV1_NL_COT': {
            'class': TraderGV1,
            'params': {'use_cot': True, 'belief_format': 'nl'}
        },
        'GV1_JSON_NOCOT': {
            'class': TraderGV1,
            'params': {'use_cot': False, 'belief_format': 'json'}
        },
        'GV1_NL_NOCOT': {
            'class': TraderGV1,
            'params': {'use_cot': False, 'belief_format': 'nl'}
        },

        # 14-17. Graph Variance 2 (Probabilistic)
        'GV2_JSON_COT': {
            'class': TraderGV2,
            'params': {'use_cot': True, 'belief_format': 'json'}
        },
        'GV2_NL_COT': {
            'class': TraderGV2,
            'params': {'use_cot': True, 'belief_format': 'nl'}
        },
        'GV2_JSON_NOCOT': {
            'class': TraderGV2,
            'params': {'use_cot': False, 'belief_format': 'json'}
        },
        'GV2_NL_NOCOT': {
            'class': TraderGV2,
            'params': {'use_cot': False, 'belief_format': 'nl'}
        },

        # 18-21. Graph Variance 3 (LLM-Designed Attributes)
        'GV3_JSON_COT': {
            'class': TraderGV3,
            'params': {'use_cot': True, 'belief_format': 'json'}
        },
        'GV3_NL_COT': {
            'class': TraderGV3,
            'params': {'use_cot': True, 'belief_format': 'nl'}
        },
        'GV3_JSON_NOCOT': {
            'class': TraderGV3,
            'params': {'use_cot': False, 'belief_format': 'json'}
        },
        'GV3_NL_NOCOT': {
            'class': TraderGV3,
            'params': {'use_cot': False, 'belief_format': 'nl'}
        },

        # 22-25. Hypothetical Minds
        'HM_JSON_COT': {
            'class': TraderHM,
            'params': {'use_cot': True, 'belief_format': 'json'}
        },
        'HM_NL_COT': {
            'class': TraderHM,
            'params': {'use_cot': True, 'belief_format': 'nl'}
        },
        'HM_JSON_NOCOT': {
            'class': TraderHM,
            'params': {'use_cot': False, 'belief_format': 'json'}
        },
        'HM_NL_NOCOT': {
            'class': TraderHM,
            'params': {'use_cot': False, 'belief_format': 'nl'}
        },
    }

    @classmethod
    def create_agent(cls, agent_type: str, tid: str, balance: float, params: dict, time: float):
        """
        Create an agent of the specified type

        Args:
            agent_type: One of the 25 agent type keys (e.g., 'LLM', 'BG_JSON_COT', 'HM_NL_NOCOT')
            tid: Trader ID
            balance: Starting balance
            params: Additional parameters (api_key, model_name, etc.)
            time: Current time

        Returns:
            Instance of the specified agent type
        """
        if agent_type not in cls.AGENT_CONFIGS:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(cls.AGENT_CONFIGS.keys())}")

        config = cls.AGENT_CONFIGS[agent_type]
        agent_class = config['class']
        agent_params = {**config['params'], **(params or {})}

        return agent_class(
            ttype=agent_type,
            tid=tid,
            balance=balance,
            params=agent_params,
            time=time
        )

    @classmethod
    def list_agent_types(cls) -> list:
        """Return list of all 25 available agent types"""
        return list(cls.AGENT_CONFIGS.keys())

    @classmethod
    def get_agent_info(cls, agent_type: str) -> dict:
        """Get configuration info for a specific agent type"""
        if agent_type not in cls.AGENT_CONFIGS:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return cls.AGENT_CONFIGS[agent_type]
