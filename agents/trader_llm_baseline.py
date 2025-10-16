#!/usr/bin/env python3
"""
Agent 1: LLM Baseline
Base LLM with market data only - no belief graph, no CoT
"""

import os
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_llm_trader import BaseLLMTrader
from .unified_prompts import PromptBuilder, BasePrompts
from BSE import Order


class TraderLLM_Baseline(BaseLLMTrader):
    """LLM trader with market data only - no belief graph"""

    def __init__(self, ttype: str, tid: str, balance: float, params: Dict[str, Any], time: float):
        super().__init__(ttype, tid, balance, params, time)

    def getorder(self, time, countdown, lob, p_eq=None, q_eq=None, demand_curve=None, supply_curve=None):
        """Generate trading order using LLM with market data only"""
        try:
            self.logger.info(f"[GETORDER] Called at time {time:.1f}, inventory={self.inventory}, balance=${self.balance:.0f}")

            if len(lob['bids']['lob']) <= 0 and len(lob['asks']['lob']) <= 0:
                self.logger.info(f"[GETORDER] Empty LOB, returning None")
                return None

            recent_prices = self.extract_recent_prices(lob, n_prices=5)
            trader_state = self.build_trader_state()
            trader_state['recent_prices'] = recent_prices

            market_context = BasePrompts.format_market_context(lob, time, trader_state)

            agent_config = {
                'use_belief_graph': False,
                'use_cot': False,
                'graph_quality': None
            }

            prompt = PromptBuilder.build_trading_prompt(
                agent_config,
                market_context,
                trader_state,
                belief_graph_data=None
            )

            decision = self.get_llm_decision(prompt, time)

            self.trading_history.append({
                'time': time,
                'decision': decision['action'],
                'reasoning': decision['reasoning'][:200]
            })

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
