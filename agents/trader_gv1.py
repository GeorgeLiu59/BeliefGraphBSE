#!/usr/bin/env python3
"""
TraderGV1: Graph Variance 1 - Discrete Sets
Configurable: use_cot (True/False), belief_format ('json' or 'nl')

Uses discrete belief sets with set elimination logic.
"""

import os
import sys
from typing import Dict, Any
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_llm_trader import BaseLLMTrader
from .unified_prompts import PromptBuilder, BasePrompts
from .belief_graph import GraphVar1
from BSE import Order


class TraderGV1(BaseLLMTrader):
    """Discrete sets belief graph trader - configurable CoT and format"""

    def __init__(self, ttype: str, tid: str, balance: float, params: Dict[str, Any], time: float):
        super().__init__(ttype, tid, balance, params, time)

        self.use_cot = params.get('use_cot', True)
        self.belief_format = params.get('belief_format', 'json')

        self.belief_graph = GraphVar1(asset_id="BSE_ASSET", model=self.model, logger=self.logger)
        self.belief_graph.add_agent(tid)

    def get_belief_data(self) -> str:
        """Get belief graph data from LLM-inferred discrete beliefs"""
        if self.belief_format == 'json':
            beliefs = {
                'tracked_agents': [],
                'belief_sets': {},
                'market_state': 'active'
            }

            for agent_id in self.belief_graph.agents:
                if agent_id != self.tid:
                    beliefs['tracked_agents'].append(agent_id)

                    # Get LLM-inferred discrete beliefs
                    llm_beliefs = self.belief_graph.discrete_beliefs[agent_id]
                    beliefs['belief_sets'][str(agent_id)] = llm_beliefs

            return json.dumps(beliefs, indent=2)
        else:
            narrative_parts = []
            for agent_id in self.belief_graph.agents:
                if agent_id != self.tid:
                    llm_beliefs = self.belief_graph.discrete_beliefs[agent_id]
                    narrative_parts.append(f"Agent {agent_id}: {llm_beliefs}")

            return "\n".join(narrative_parts) if narrative_parts else "No agents observed yet."

    def getorder(self, time, countdown, lob, p_eq=None, q_eq=None, demand_curve=None, supply_curve=None):
        try:
            self.logger.info(f"[GETORDER] Called at time {time:.1f}, job={self.job}, balance=${self.balance:.0f}")

            if len(lob['bids']['lob']) <= 0 and len(lob['asks']['lob']) <= 0:
                self.logger.info(f"[GETORDER] Empty LOB, returning None")
                return None

            recent_prices = self.extract_recent_prices(lob, n_prices=5)
            trader_state = self.build_trader_state()
            trader_state['recent_prices'] = recent_prices

            market_context = BasePrompts.format_market_context(lob, time, trader_state)
            belief_data = self.get_belief_data()

            agent_config = {
                'use_belief_graph': True,
                'belief_format': self.belief_format,
                'use_cot': self.use_cot,
                'graph_quality': 'basic',
                'job': self.job
            }

            prompt = PromptBuilder.build_trading_prompt(agent_config, market_context, trader_state, belief_graph_data=belief_data)
            decision = self.get_llm_decision(prompt, time)

            if decision['action'] == 'WAIT':
                return None

            return Order(self.tid, 'Bid' if self.job == 'Buy' else 'Ask', decision['price'], 1, time, lob['QID'])
        except Exception as e:
            self.logger.error(f"[GETORDER-ERROR] Failed at time {time:.1f}: {e}", exc_info=True)
            return None

    def respond(self, time, lob, trade, verbose):
        """Update belief graph when market events occur"""
        events_processed = self.process_and_log_market_events(time, lob, trade)
        self.log_belief_graph_update(time, events_processed, lob)