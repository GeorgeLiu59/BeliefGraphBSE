#!/usr/bin/env python3
"""
TraderHM: Hypothetical-Minds Trader
Configurable: use_cot (True/False), belief_format ('json' or 'nl')

Uses HypothesisScaffold for opponent modeling with Rescorla-Wagner learning.
"""

import os
import sys
from typing import Dict, Any
import json
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_llm_trader import BaseLLMTrader
from .unified_prompts import BasePrompts
from .belief_graph import BeliefGraph
from .hypothesis_scaffold import HypothesisScaffold
from BSE import Order


class TraderHM(BaseLLMTrader):
    """Hypothetical-Minds trader - configurable CoT and format"""

    def __init__(self, ttype: str, tid: str, balance: float, params: Dict[str, Any], time: float):
        super().__init__(ttype, tid, balance, params, time)

        self.use_cot = params.get('use_cot', True)
        self.belief_format = params.get('belief_format', 'json')

        self.belief_graph = BeliefGraph(asset_id="BSE_ASSET")
        self.belief_graph.add_agent(tid)

        self.hypothesis_scaffold = HypothesisScaffold(
            trader_id=tid,
            model=self.model,
            logger=self.logger,
            belief_graph=self.belief_graph
        )

        self.attributes = {
            'learning_rate': 0.6,
            'hypothesis_threshold': 0.3,
            'exploration_rate': 0.5,
            'adaptability': 0.7,
            'strategic_depth': 0.8
        }

        if self.tid in self.belief_graph.nodes:
            agent_node = self.belief_graph.nodes[self.tid]
            agent_node.strategy_type = 'Hypothetical-Minds'

        self.logger.info(f"[HM-INIT] {tid}: Initialized with HypothesisScaffold")


    def get_belief_data(self) -> str:
        """Get belief graph data in configured format"""
        if self.belief_format == 'json':
            good_hypos = self.hypothesis_scaffold.get_good_hypotheses_data()

            beliefs = {
                'tracked_agents': [str(aid) for aid in self.belief_graph.agents if aid != self.tid],
                'opponent_hypotheses_count': len(self.hypothesis_scaffold.opponent_hypotheses),
                'good_hypotheses_count': len(good_hypos),
                'good_hypotheses': good_hypos,
                'market_sentiment': 'active',
                'my_attributes': self.attributes
            }
            return json.dumps(beliefs, indent=2)
        else:
            narrative_parts = [f"My trading attributes: {self.attributes}"]

            hypothesis_context = self.hypothesis_scaffold.get_good_hypotheses_context()
            if hypothesis_context:
                narrative_parts.append(hypothesis_context)

            narrative_parts.append(f"\nTracking {len(self.belief_graph.agents)} agents")
            for agent_id in self.belief_graph.agents:
                if agent_id != self.tid:
                    narrative_parts.append(f"  - Agent {agent_id}")

            return "\n".join(narrative_parts) if narrative_parts else "No agents observed yet."

    def getorder(self, time, countdown, lob, p_eq=None, q_eq=None, demand_curve=None, supply_curve=None):
        try:
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

            from .unified_prompts import PromptBuilder
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
        """Update belief graph and evaluate hypotheses - WITH THROTTLE"""
        from .belief_graph import MarketEvent, EventType

        # THROTTLE CHECK - same as other agents
        if (time - self.last_belief_update_time) < self.belief_update_interval:
            return

        self.last_belief_update_time = time
        self.logger.info(f"[BELIEF-THROTTLE] {self.tid}: Running belief graph update at time {time:.1f}")

        self.logger.debug(f"[HM-RESPOND] Called at time {time:.1f}")
        events_processed = 0

        if trade and 'party1' in trade and 'party2' in trade:
            buyer_id = trade['party1']
            seller_id = trade['party2']
            trade_price = trade['price']

            self.logger.debug(f"[HM-TRADE] Buyer: {buyer_id}, Seller: {seller_id}, Price: ${trade_price}")

            recent_prices = self.extract_recent_prices(lob, n_prices=5)
            best_bid = lob.get('bids', {}).get('best')
            best_ask = lob.get('asks', {}).get('best')

            last_trade_price = None
            if lob.get('tape') and len(lob['tape']) > 0:
                for i in range(len(lob['tape']) - 1, -1, -1):
                    if lob['tape'][i].get('type') == 'Trade':
                        last_trade_price = lob['tape'][i]['price']
                        break

            self.logger.info(f"[LOB-DATA-HM] best_bid: {best_bid}, best_ask: {best_ask}, last_trade_price: {last_trade_price}")

            market_context = {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'last_trade_price': last_trade_price,
                'recent_prices': recent_prices
            }

            opponent_trader_id = None
            if buyer_id != self.tid:
                opponent_trader_id = buyer_id
                event = MarketEvent(
                    event_id=f"trade_{buyer_id}_{time}",
                    event_type=EventType.TRADE,
                    timestamp=time,
                    agent_id=buyer_id,
                    price=trade_price,
                    quantity=1,
                    counterparty_id=seller_id
                )
                if buyer_id not in self.belief_graph.agents:
                    self.belief_graph.add_agent(buyer_id)
                    self.logger.info(f"[HM-NEW-AGENT] Added buyer {buyer_id}")
                self.belief_graph.update_beliefs(event)
                events_processed += 1

                action_data = {
                    'price': trade_price,
                    'time': time,
                    'event_type': 'trade',
                    'quantity': 1,
                    'opponent_id': buyer_id
                }
                self.hypothesis_scaffold.observe_opponent_action(buyer_id, action_data, market_context)

            if seller_id != self.tid:
                opponent_trader_id = seller_id
                event = MarketEvent(
                    event_id=f"trade_{seller_id}_{time}",
                    event_type=EventType.TRADE,
                    timestamp=time,
                    agent_id=seller_id,
                    price=trade_price,
                    quantity=1,
                    counterparty_id=buyer_id
                )
                if seller_id not in self.belief_graph.agents:
                    self.belief_graph.add_agent(seller_id)
                    self.logger.info(f"[HM-NEW-AGENT] Added seller {seller_id}")
                self.belief_graph.update_beliefs(event)
                events_processed += 1

                action_data = {
                    'price': trade_price,
                    'time': time,
                    'event_type': 'trade',
                    'quantity': 1,
                    'opponent_id': seller_id
                }
                self.hypothesis_scaffold.observe_opponent_action(seller_id, action_data, market_context)

            if opponent_trader_id:
                actual_action = {
                    'price': trade_price,
                    'time': time,
                    'event_type': 'trade',
                    'quantity': 1,
                    'opponent_id': opponent_trader_id
                }
                try:
                    asyncio.get_running_loop()
                    self.logger.info(f"[HYP-SKIP] {self.tid}: Skipping async evaluation (in event loop)")
                except RuntimeError:
                    asyncio.run(self.hypothesis_scaffold.evaluate_hypotheses(
                        opponent_trader_id, actual_action, market_context
                    ))

        if lob['bids']['n'] > 0:
            for bid in lob['bids']['lob']:
                if bid[0] != self.tid:
                    event = MarketEvent(
                        event_id=f"bid_{bid[0]}_{time}",
                        event_type=EventType.BID,
                        timestamp=time,
                        agent_id=bid[0],
                        price=bid[1],
                        quantity=1
                    )
                    if bid[0] not in self.belief_graph.agents:
                        self.belief_graph.add_agent(bid[0])
                    self.belief_graph.update_beliefs(event)
                    events_processed += 1

        if lob['asks']['n'] > 0:
            for ask in lob['asks']['lob']:
                if ask[0] != self.tid:
                    event = MarketEvent(
                        event_id=f"ask_{ask[0]}_{time}",
                        event_type=EventType.ASK,
                        timestamp=time,
                        agent_id=ask[0],
                        price=ask[1],
                        quantity=1
                    )
                    if ask[0] not in self.belief_graph.agents:
                        self.belief_graph.add_agent(ask[0])
                    self.belief_graph.update_beliefs(event)
                    events_processed += 1

        self.log_belief_graph_update(time, events_processed, lob)
