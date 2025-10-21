#!/usr/bin/env python3
"""
TraderPG: Perfect Graph Trader
Configurable: use_cot (True/False), belief_format ('json' or 'nl')
"""

import os
import sys
from typing import Dict, Any
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_llm_trader import BaseLLMTrader
from .unified_prompts import PromptBuilder, BasePrompts
from .belief_graph import BeliefGraph, PerfectBeliefGraph
from BSE import Order


class TraderPG(BaseLLMTrader):
    """Perfect belief graph trader - configurable CoT and format"""

    def __init__(self, ttype: str, tid: str, balance: float, params: Dict[str, Any], time: float):
        super().__init__(ttype, tid, balance, params, time)

        self.use_cot = params.get('use_cot', True)
        self.belief_format = params.get('belief_format', 'json')

        # Deferred initialization - wait for traders_dict
        self.belief_graph = None
        self.traders_dict = None

    def set_traders_dict(self, traders_dict: Dict[str, Any]):
        """BSE calls this to give PG access to all traders (CHEAT MODE)"""
        self.traders_dict = traders_dict
        # NOW create PerfectBeliefGraph with cheat access
        self.belief_graph = PerfectBeliefGraph(asset_id="BSE_ASSET", traders_dict=traders_dict)
        self.belief_graph.add_agent(self.tid)

    def get_belief_data(self) -> str:
        """Get belief graph data in configured format - PERFECT KNOWLEDGE, no aggressiveness inference"""
        if not self.belief_graph:
            return json.dumps({"error": "Perfect graph not initialized yet"}) if self.belief_format == 'json' else "Perfect graph not initialized yet"

        if self.belief_format == 'json':
            beliefs = {
                'strategy_beliefs': {},
                'perfect_knowledge': True,
                'market_sentiment': 'unknown',
                'risk_assessment': 'unknown'
            }
            for agent_id in self.belief_graph.nodes:
                if agent_id != self.tid and agent_id != "BSE_ASSET":
                    beliefs['strategy_beliefs'][agent_id] = self.belief_graph.get_beliefs(agent_id)
            return json.dumps(beliefs, indent=2)
        else:
            traders_info = []
            for agent_id in self.belief_graph.nodes:
                if agent_id != self.tid and agent_id != "BSE_ASSET":
                    agent_beliefs = self.belief_graph.get_agent_beliefs(agent_id)
                    val_est = agent_beliefs.get('valuation_estimate')
                    strategy = agent_beliefs.get('strategy_type', 'unknown')

                    if val_est and isinstance(val_est, (int, float)):
                        traders_info.append((agent_id, val_est, strategy))

            if not traders_info:
                return "No other traders observed yet."

            traders_info.sort(key=lambda x: x[1])

            parts = ["PERFECT KNOWLEDGE - OTHER TRADERS' EXACT VALUATIONS & STRATEGIES:"]
            for agent_id, val, strategy in traders_info:
                strategy_display = strategy if strategy else 'unknown'
                parts.append(f"  {agent_id}: ${val:.0f} (strategy: {strategy_display})")

            lowest = traders_info[0]
            highest = traders_info[-1]
            parts.append(f"\nKEY INSIGHTS:")
            parts.append(f"  Lowest valuation: {lowest[0]} at ${lowest[1]:.0f}")
            parts.append(f"  Highest valuation: {highest[0]} at ${highest[1]:.0f}")
            parts.append(f"  You have perfect info - exploit these exact valuations!")

            return "\n".join(parts)

    def getorder(self, time, countdown, lob, p_eq=None, q_eq=None, demand_curve=None, supply_curve=None):
        try:
            self.logger.info(f"[GETORDER] Called at time {time:.1f}, inventory={self.inventory}, balance=${self.balance:.0f}")

            if len(lob['bids']['lob']) <= 0 and len(lob['asks']['lob']) <= 0:
                self.logger.info(f"[GETORDER] Empty LOB, returning None")
                return None

            recent_prices = self.extract_recent_prices(lob, n_prices=5)
            trader_state = self.build_trader_state()
            trader_state['recent_prices'] = recent_prices
            trader_state['time'] = time  # Add time to trader_state for the context

            market_context = BasePrompts.format_market_context(lob, trader_state)
            belief_data = self.get_belief_data()

            agent_config = {
                'use_belief_graph': True,
                'belief_format': self.belief_format,
                'use_cot': self.use_cot,
                'graph_quality': 'perfect'
            }

            prompt = PromptBuilder.build_trading_prompt(agent_config, market_context, trader_state, belief_graph_data=belief_data)
            decision = self.get_llm_decision(prompt, time)

            if decision['action'] == 'BUY':
                if decision['price'] > self.balance or self.inventory >= 10:
                    return None
                return Order(self.tid, 'Bid', decision['price'], 1, time, lob['QID'])
            elif decision['action'] == 'SELL':
                if self.inventory <= 0:
                    return None
                return Order(self.tid, 'Ask', decision['price'], 1, time, lob['QID'])

            return None
        except Exception as e:
            self.logger.error(f"[GETORDER-ERROR] Failed at time {time:.1f}: {e}", exc_info=True)
            return None

    def respond(self, time, lob, trade, verbose):
        """Update belief graph when market events occur"""
        events_processed = self.process_and_log_market_events(time, lob, trade)
        self.log_belief_graph_update(time, events_processed, lob)
