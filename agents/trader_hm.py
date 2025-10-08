#!/usr/bin/env python3
"""
TraderHM: Hypothetical-Minds Trader
Configurable: use_cot (True/False), belief_format ('json' or 'nl')

Uses hypothesis generation and Rescorla-Wagner learning.
"""

import os
import sys
from typing import Dict, Any
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_llm_trader import BaseLLMTrader
from unified_prompts import BasePrompts, HypotheticalMindPrompts
from belief_graph import BeliefGraph
from BSE import Order


class TraderHM(BaseLLMTrader):
    """Hypothetical-Minds trader - configurable CoT and format"""

    def __init__(self, ttype: str, tid: str, balance: float, params: Dict[str, Any], time: float):
        super().__init__(ttype, tid, balance, params, time)

        self.use_cot = params.get('use_cot', True)
        self.belief_format = params.get('belief_format', 'json')

        self.belief_graph = BeliefGraph(asset_id="BSE_ASSET")
        self.belief_graph.add_agent(tid)

        self.opponent_hypotheses = {}
        self.interaction_history = []
        self.max_interaction_history = 20
        self.good_hypothesis_found = False
        self.interaction_num = 0

        self.alpha = 0.6
        self.correct_guess_reward = 1
        self.good_hypothesis_thr = 0.3
        self.top_k = 3

        self.attributes = {
            'aggressiveness': 0.6,
            'learning_rate': 0.6,
            'hypothesis_threshold': 0.3,
            'exploration_rate': 0.5,
            'adaptability': 0.7,
            'strategic_depth': 0.8
        }

        if self.tid in self.belief_graph.nodes:
            agent_node = self.belief_graph.nodes[self.tid]
            agent_node.aggressiveness_score = self.attributes['aggressiveness']
            agent_node.strategy_type = 'Hypothetical-Minds'

    def generate_hypotheses(self, market_context: str, target_trader_id: str) -> Dict:
        """Generate hypothesis about specific opponent's strategy"""
        if not self.model:
            return {}

        prompt = HypotheticalMindPrompts.hypothesis_generation_prompt(
            market_context,
            self.interaction_history[-5:] if self.interaction_history else []
        )

        response = self.model.generate_content(
            prompt,
            generation_config=self.model._generation_config
        )

        try:
            hypotheses_data = json.loads(response.text)
            hypotheses_list = hypotheses_data.get('hypotheses', [])

            if hypotheses_list:
                hypothesis = hypotheses_list[0]
                return {
                    'target_trader_id': target_trader_id,
                    'possible_other_player_strategy': hypothesis.get('description', 'Unknown strategy'),
                    'value': 0,
                    'other_player_next_action': {}
                }
            return {}
        except:
            return {}

    def get_belief_data(self) -> str:
        """Get belief graph data in configured format"""
        if self.belief_format == 'json':
            beliefs = {
                'tracked_agents': [str(aid) for aid in self.belief_graph.agents if aid != self.tid],
                'opponent_hypotheses_count': len(self.opponent_hypotheses),
                'good_hypotheses_count': len([h for h in self.opponent_hypotheses.values() if h.get('value', 0) > self.good_hypothesis_thr]),
                'market_sentiment': 'active',
                'my_attributes': self.attributes
            }
            return json.dumps(beliefs, indent=2)
        else:
            narrative_parts = [f"My trading attributes: {self.attributes}"]

            good_hypotheses = self.gather_all_good_hypotheses()
            if good_hypotheses:
                narrative_parts.append("\nProven opponent models (good hypotheses):")
                for trader_id, hyps in good_hypotheses.items():
                    narrative_parts.append(f"  {trader_id}:")
                    for h in hyps[:self.top_k]:
                        narrative_parts.append(f"    - {h['strategy'][:100]}... (value: {h['value']:.2f})")

            narrative_parts.append(f"\nTracking {len(self.belief_graph.agents)} agents")
            for agent_id in self.belief_graph.agents:
                if agent_id != self.tid:
                    narrative_parts.append(f"  - Agent {agent_id}")

            return "\n".join(narrative_parts) if narrative_parts else "No agents observed yet."

    def evaluate_predicted_behavior(self, hypothesis_key: int, target_trader_id: str) -> str:
        """Generate prompt to evaluate if hypothesis prediction matches actual behavior"""
        hypothesis = self.opponent_hypotheses.get(hypothesis_key, {})
        hypothesis_strategy = hypothesis.get('possible_other_player_strategy', 'No strategy stored')

        latest_interaction = self.interaction_history[-1] if self.interaction_history else {}
        actual_action = latest_interaction.get('action', 'Unknown')
        actual_price = latest_interaction.get('price', 'Unknown')

        user_message = f"""
        Does {target_trader_id}'s current action match our hypothesis about their behavior?

        === OUR HYPOTHESIS ===
        We predicted: {hypothesis_strategy}

        === WHAT THEY ACTUALLY DID ===
        Action: {actual_action}
        Price: {actual_price}

        Does their actual trading behavior align with our hypothesis?
        Answer with a simple Yes or No, followed by brief reasoning.

        ```python
        {{
          'evaluate_predicted_behavior': True
        }}
        ```
        """
        return user_message

    def evaluate_opponent_hypotheses(self, target_trader_id: str):
        """Evaluate and update opponent strategy hypotheses using Rescorla-Wagner learning"""
        trader_hypotheses = {k: v for k, v in self.opponent_hypotheses.items()
                           if v.get('target_trader_id') == target_trader_id}

        if not trader_hypotheses:
            return

        latest_key = max(trader_hypotheses.keys())
        sorted_keys = sorted([key for key in trader_hypotheses if key != latest_key],
                           key=lambda x: trader_hypotheses[x].get('value', 0),
                           reverse=True)
        keys2eval = sorted_keys[:self.top_k] + [latest_key]

        valid_keys2eval = []
        for key in keys2eval:
            if 'other_player_next_action' in trader_hypotheses[key]:
                valid_keys2eval.append(key)

        if not valid_keys2eval:
            return

        self.good_hypothesis_found = False

        for key in valid_keys2eval:
            eval_prompt = self.evaluate_predicted_behavior(key, target_trader_id)

            if not self.model:
                continue

            response = self.model.generate_content(
                eval_prompt,
                generation_config=self.model._generation_config
            )

            try:
                pred_label = self._extract_eval_result(response.text)

                old_value = self.opponent_hypotheses[key].get('value', 0)
                if pred_label.get('evaluate_predicted_behavior'):
                    prediction_error = self.correct_guess_reward - old_value
                else:
                    prediction_error = -self.correct_guess_reward - old_value

                self.opponent_hypotheses[key]['value'] = old_value + self.alpha * prediction_error
                new_value = self.opponent_hypotheses[key]['value']

                if new_value > self.good_hypothesis_thr:
                    self.good_hypothesis_found = True

            except Exception as e:
                continue

    def _extract_eval_result(self, response_text: str) -> Dict:
        """Extract evaluation result from LLM response"""
        response_lower = response_text.lower()

        if 'true' in response_lower or 'yes' in response_lower or 'aligns' in response_lower:
            return {'evaluate_predicted_behavior': True}
        else:
            return {'evaluate_predicted_behavior': False}

    def gather_all_good_hypotheses(self) -> Dict[str, Any]:
        """Gather all hypotheses above the good threshold, grouped by target trader"""
        good_hypotheses_by_trader = {}

        for key, hypothesis in self.opponent_hypotheses.items():
            if hypothesis.get('value', 0) > self.good_hypothesis_thr:
                target_trader_id = hypothesis.get('target_trader_id')
                if target_trader_id:
                    if target_trader_id not in good_hypotheses_by_trader:
                        good_hypotheses_by_trader[target_trader_id] = []

                    good_hypotheses_by_trader[target_trader_id].append({
                        'key': key,
                        'value': hypothesis.get('value', 0),
                        'strategy': hypothesis.get('possible_other_player_strategy', 'Unknown'),
                        'target_trader': target_trader_id,
                        'prediction': hypothesis.get('other_player_next_action', {})
                    })

        for trader_id in good_hypotheses_by_trader:
            good_hypotheses_by_trader[trader_id].sort(key=lambda x: x['value'], reverse=True)

        return good_hypotheses_by_trader

    def generate_strategic_order_with_all_good_hypotheses(self, lob: Dict[str, Any], time: float, countdown: float) -> str:
        """Generate strategic order prompt using all good hypotheses"""
        good_hypotheses_by_trader = self.gather_all_good_hypotheses()

        if not good_hypotheses_by_trader:
            return None

        market_context = {
            'time': time,
            'countdown': countdown,
            'best_bid': lob.get('bids', {}).get('best', 'None'),
            'best_ask': lob.get('asks', {}).get('best', 'None'),
            'your_balance': self.balance,
            'job': self.job
        }

        recent_interactions = self.interaction_history[-5:] if self.interaction_history else []

        prompt = f"""
        {BasePrompts.MARKET_FUNDAMENTALS}

        STRATEGIC ORDER DECISION WITH ALL PROVEN OPPONENT MODELS

        === MARKET CONTEXT ===
        {market_context}

        === YOUR ATTRIBUTES ===
        {self.attributes}

        === ALL GOOD HYPOTHESES (Value > {self.good_hypothesis_thr}) ===
        {good_hypotheses_by_trader}

        === RECENT INTERACTIONS ===
        {recent_interactions}

        TASK: Generate optimal order price using ALL proven opponent models and current market state.

        You must respond with ONLY valid JSON in this format:
        {{
          "action": "BID" or "ASK" or "WAIT",
          "price": 150,
          "reasoning": "Strategic reasoning using opponent models",
          "confidence": 0.8
        }}
        """

        return prompt

    def generate_vanilla_order_prompt(self, lob: Dict[str, Any], time: float, countdown: float) -> str:
        """Generate simple order prompt when no good hypotheses exist"""
        market_info = {
            'time': time,
            'countdown': countdown,
            'best_bid': lob.get('bids', {}).get('best', 'None'),
            'best_ask': lob.get('asks', {}).get('best', 'None'),
            'balance': self.balance,
            'job': self.job
        }

        prompt = f"""
        {BasePrompts.MARKET_FUNDAMENTALS}

        SIMPLE TRADING DECISION - Basic Market Analysis

        === CURRENT MARKET STATE ===
        {market_info}

        === YOUR ATTRIBUTES ===
        {self.attributes}

        TASK: Make a simple trading decision based on current market conditions.
        You're still learning about opponents - no proven hypotheses yet.

        You must respond with ONLY valid JSON in this format:
        {{
          "action": "BID" or "ASK" or "WAIT",
          "price": 150,
          "reasoning": "Simple market analysis reasoning",
          "confidence": 0.7
        }}
        """

        return prompt

    def getorder(self, time, countdown, lob, p_eq=None, q_eq=None, demand_curve=None, supply_curve=None):
        if len(lob['bids']['lob']) <= 0 and len(lob['asks']['lob']) <= 0:
            return None

        recent_prices = self.extract_recent_prices(lob, n_prices=5)
        trader_state = self.build_trader_state()
        trader_state['recent_prices'] = recent_prices

        market_context = BasePrompts.format_market_context(lob, time, trader_state)

        if len(self.interaction_history) % 5 == 0:
            for agent_id in self.belief_graph.agents:
                if agent_id != self.tid:
                    hypothesis = self.generate_hypotheses(market_context, agent_id)
                    if hypothesis:
                        hypothesis_key = self.interaction_num + len(self.opponent_hypotheses)
                        self.opponent_hypotheses[hypothesis_key] = hypothesis

        if self.good_hypothesis_found:
            strategic_prompt = self.generate_strategic_order_with_all_good_hypotheses(lob, time, countdown)
            if strategic_prompt:
                decision = self.get_llm_decision(strategic_prompt)
            else:
                vanilla_prompt = self.generate_vanilla_order_prompt(lob, time, countdown)
                decision = self.get_llm_decision(vanilla_prompt)
        else:
            vanilla_prompt = self.generate_vanilla_order_prompt(lob, time, countdown)
            decision = self.get_llm_decision(vanilla_prompt)

        self.interaction_history.append({
            'time': time,
            'action': decision['action'],
            'price': decision.get('price', 0),
            'reasoning': decision['reasoning'][:100]
        })

        if len(self.interaction_history) > self.max_interaction_history:
            self.interaction_history = self.interaction_history[-self.max_interaction_history:]

        if decision['action'] == 'WAIT':
            return None

        return Order(self.tid, 'Bid' if self.job == 'Buy' else 'Ask', decision['price'], 1, time, lob['QID'])

    def respond(self, time, lob, trade, verbose):
        """Update belief graph when market events occur and evaluate hypotheses"""
        from belief_graph import MarketEvent, EventType

        self.logger.debug(f"[HM-RESPOND] Called at time {time:.1f}")
        events_processed = 0

        if trade and 'party1' in trade and 'party2' in trade:
            buyer_id = trade['party1']
            seller_id = trade['party2']
            trade_price = trade['price']

            self.logger.debug(f"[HM-TRADE] Buyer: {buyer_id}, Seller: {seller_id}, Price: ${trade_price}")

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

            if opponent_trader_id:
                self.interaction_num += 1
                self.logger.info(f"[HM-INTERACTION] #{self.interaction_num} with {opponent_trader_id}")

                for key in self.opponent_hypotheses:
                    if self.opponent_hypotheses[key].get('target_trader_id') == opponent_trader_id:
                        self.opponent_hypotheses[key]['other_player_next_action'] = {
                            'predicted_price': trade_price
                        }

                self.logger.debug(f"[HM-EVAL] Evaluating hypotheses for {opponent_trader_id}")
                self.evaluate_opponent_hypotheses(opponent_trader_id)

                good_hyps = self.gather_all_good_hypotheses()
                if opponent_trader_id in good_hyps and good_hyps[opponent_trader_id]:
                    best_hypothesis = good_hyps[opponent_trader_id][0]
                    self.logger.info(f"[HM-BEST-HYP] For {opponent_trader_id}: {best_hypothesis['strategy'][:100]} (value: {best_hypothesis['value']:.3f})")
                    if opponent_trader_id in self.belief_graph.nodes:
                        opponent_node = self.belief_graph.nodes[opponent_trader_id]
                        opponent_node.strategy_type = best_hypothesis['strategy'][:100]

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

        self.log_belief_graph_update(time, events_processed)
