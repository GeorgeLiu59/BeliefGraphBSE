#!/usr/bin/env python3
"""
TraderGV3: Graph Variance 3 - LLM-Designed Attributes
Configurable: use_cot (True/False), belief_format ('json' or 'nl')

NO HARDCODED ATTRIBUTES. LLM designs them dynamically.
"""

import os
import sys
from typing import Dict, Any
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_llm_trader import BaseLLMTrader
from .unified_prompts import PromptBuilder, BasePrompts, AdaptiveAttributePrompts, PromptParser
from .belief_graph import GraphVar3
from BSE import Order


class TraderGV3(BaseLLMTrader):
    """LLM-designed attributes trader - configurable CoT and format"""

    def __init__(self, ttype: str, tid: str, balance: float, params: Dict[str, Any], time: float):
        super().__init__(ttype, tid, balance, params, time)

        self.use_cot = params.get('use_cot', True)
        self.belief_format = params.get('belief_format', 'json')

        self.belief_graph = GraphVar3(asset_id="BSE_ASSET", model=self.model, logger=self.logger)
        # Note: We don't add self to our own belief graph - it only tracks OTHER agents

        self.attributes = None
        self.attributes_initialized = False

        self.adaptation_enabled = params.get('adaptation_enabled', True)
        self.adaptation_interval = params.get('adaptation_interval', 10)
        self.last_adaptation_check = 0

    def initialize_attributes(self, market_context: Dict[str, Any] = None):
        """LLM designs its own attributes dynamically"""
        if self.attributes_initialized:
            return

        market_ctx = market_context

        prompt = AdaptiveAttributePrompts.design_attributes_prompt(market_ctx)

        self.logger.info("=== ATTRIBUTE DESIGN PROMPT ===")
        self.logger.info("="*80)
        self.logger.info(prompt)
        self.logger.info("="*80)

        response = self.model.generate_content(
            prompt,
            generation_config=self.model._generation_config
        )

        self.logger.info("=== ATTRIBUTE DESIGN RESPONSE ===")
        self.logger.info(response.text.strip())
        self.logger.info("="*80)

        self.attributes = PromptParser.parse_attribute_design(response.text)
        self.attributes_initialized = True

        self.logger.info(f"=== DESIGNED ATTRIBUTES ===")
        for key, value in self.attributes.items():
            if key == 'reasoning':
                self.logger.info(f"{key.replace('_', ' ').title()}: {str(value)[:200]}")
            else:
                self.logger.info(f"{key.replace('_', ' ').title()}: {value}")
        self.logger.info("="*80)

        # Note: We don't sync attributes to belief graph because our belief graph only tracks OTHER agents
        # Our own attributes (self.attributes) represent our trading personality
        # The belief graph tracks what we believe about competitor behavior

    def check_and_adapt_attributes(self, market_context: Dict[str, Any] = None):
        """Adapt attributes based on performance"""
        if not self.adaptation_enabled:
            return

        if self.n_trades < self.last_adaptation_check + self.adaptation_interval:
            return

        if not self.model:
            return

        performance_metrics = {
            'profit': self.total_profit,
            'win_rate': 0.5,
            'trade_count': self.n_trades
        }

        market_ctx = market_context

        prompt = AdaptiveAttributePrompts.adapt_attributes_prompt(
            self.attributes,
            performance_metrics,
            market_ctx
        )

        self.logger.info("=== ATTRIBUTE ADAPTATION PROMPT ===")
        self.logger.info("="*80)
        self.logger.info(prompt)
        self.logger.info("="*80)

        response = self.model.generate_content(
            prompt,
            generation_config=self.model._generation_config
        )

        self.logger.info("=== ATTRIBUTE ADAPTATION RESPONSE ===")
        self.logger.info(response.text.strip())
        self.logger.info("="*80)

        new_attributes = PromptParser.parse_attribute_design(response.text)
        old_attributes = self.attributes.copy()
        self.attributes = new_attributes
        self.last_adaptation_check = self.n_trades

        self.logger.info(f"=== ATTRIBUTES ADAPTED (Trade #{self.n_trades}) ===")
        self.logger.info(f"OLD -> NEW:")
        all_keys = set(old_attributes.keys()) | set(new_attributes.keys())
        for key in sorted(all_keys):
            old_val = old_attributes.get(key)
            new_val = new_attributes.get(key)
            if key == 'reasoning':
                self.logger.info(f"{key.replace('_', ' ').title()}: {str(new_val)[:200]}")
            elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                change = f" ({new_val - old_val:+.2f})"
                self.logger.info(f"{key.replace('_', ' ').title()}: {old_val} -> {new_val}{change}")
            else:
                self.logger.info(f"{key.replace('_', ' ').title()}: {old_val} -> {new_val}")
        self.logger.info("="*80)

        # Note: Adapted attributes affect our OWN trading decisions, not belief graph
        # Belief graph tracks what we observe about OTHER agents

    def get_belief_data(self) -> str:
        """Get belief graph data in configured format"""
        if self.belief_format == 'json':
            beliefs = {
                'my_attributes': self.attributes,
                'competitor_belief_traits': {}
            }
            for agent_id in self.belief_graph.agents:
                if agent_id != self.tid:
                    agent_beliefs = self.belief_graph.get_agent_beliefs(agent_id)
                    beliefs['competitor_belief_traits'][agent_id] = agent_beliefs.get('attributes', {})
            return json.dumps(beliefs, indent=2)
        else:
            narrative_parts = [f"My trading attributes: {self.attributes}"]
            for agent_id in self.belief_graph.agents:
                if agent_id != self.tid:
                    agent_beliefs = self.belief_graph.get_agent_beliefs(agent_id)
                    belief_traits = agent_beliefs['attributes']
                    narrative_parts.append(f"Agent {agent_id}: {belief_traits}")
            return "\n".join(narrative_parts) if narrative_parts else "No agents observed yet."

    def getorder(self, time, countdown, lob, p_eq=None, q_eq=None, demand_curve=None, supply_curve=None):
        try:
            if not self.attributes_initialized:
                self.initialize_attributes()

            self.check_and_adapt_attributes()

            self.logger.info(f"[GETORDER] Called at time {time:.1f}, inventory={self.inventory}, balance=${self.balance:.0f}")

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
                'graph_quality': 'basic'
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