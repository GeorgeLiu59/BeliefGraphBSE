#!/usr/bin/env python3
"""
Hypothesis Scaffolding System for Opponent Strategy Modeling

Provides clean, orthogonal layer for hypothesis generation, evaluation,
and Rescorla-Wagner learning. Extracted from DEL.py for reuse.
"""

import asyncio
from typing import Dict, Any, Optional


class HypothesisScaffold:
    """
    Modular hypothesis scaffolding system for opponent strategy modeling.

    This class provides a clean, orthogonal layer on top of existing trading logic
    to generate, evaluate, and utilize hypotheses about opponent strategies.
    """

    def __init__(self, trader_id, model=None, logger=None, belief_graph=None):
        """Initialize the hypothesis scaffolding system"""
        self.trader_id = trader_id
        self.model = model
        self.logger = logger
        self.belief_graph = belief_graph

        self.opponent_hypotheses = {}
        self.hypothesis_counter = 0

        self.alpha = 0.6
        self.correct_guess_reward = 1
        self.good_hypothesis_thr = 0.3
        self.top_k = 3
        self.max_hypotheses_per_trader = 10

        if self.logger:
            self.logger.info(f"[HYP-SCAFFOLD] {trader_id}: Initialized hypothesis scaffolding system")

    def observe_opponent_action(self, opponent_id, action_data, market_context):
        """Create hypothesis about opponent strategy when observing their action"""
        if not self.model:
            return

        hypothesis_prompt = self._create_hypothesis_prompt(opponent_id, action_data, market_context)

        if self.logger:
            self.logger.debug(f"[HYP-CREATE-PROMPT] {self.trader_id}:")
            self.logger.debug("="*80)
            self.logger.debug(hypothesis_prompt)
            self.logger.debug("="*80)

        response = self.model.generate_content(
            hypothesis_prompt,
            generation_config=self.model._generation_config
        )

        strategy_text = response.text.strip()

        if self.logger:
            self.logger.debug(f"[HYP-CREATE-RESPONSE] {self.trader_id}:")
            self.logger.debug("-"*50)
            self.logger.debug(strategy_text)
            self.logger.debug("-"*50)

        self.hypothesis_counter += 1
        hypothesis_id = f"{opponent_id}_h{self.hypothesis_counter}"

        self.opponent_hypotheses[hypothesis_id] = {
            'target_trader': opponent_id,
            'possible_other_player_strategy': strategy_text,
            'value': 0.0,
            'created_time': action_data.get('time', 0),
            'other_player_next_action': {}
        }

        if self.logger:
            self.logger.info(f"[HYP-CREATE] {self.trader_id}: Created hypothesis {hypothesis_id} for {opponent_id}")
            self.logger.info(f"  Strategy: {strategy_text}")

        self._prune_hypotheses(opponent_id)

    async def evaluate_hypotheses(self, opponent_id, actual_action, market_context):
        """Batch evaluate top K + latest hypotheses with Rescorla-Wagner"""
        if not self.model:
            return

        trader_hypotheses = {
            h_id: h_data for h_id, h_data in self.opponent_hypotheses.items()
            if h_data['target_trader'] == opponent_id
        }

        if not trader_hypotheses:
            return

        latest_key = max(trader_hypotheses.keys()) if trader_hypotheses else None
        if not latest_key:
            return

        sorted_keys = sorted([key for key in trader_hypotheses if key != latest_key],
                           key=lambda x: trader_hypotheses[x]['value'],
                           reverse=True)
        keys2eval = sorted_keys[:self.top_k] + [latest_key]

        good_hypothesis_found = False

        valid_keys2eval = []
        user_messages = []

        for key in keys2eval:
            strategy_text = trader_hypotheses[key].get('possible_other_player_strategy', '')
            if strategy_text:
                evaluation_prompt = self._create_evaluation_prompt(
                    strategy_text, actual_action, market_context
                )

                if self.logger:
                    self.logger.info(f"[HYP-EVAL-PROMPT] {self.trader_id} for {key}:")
                    self.logger.info("="*80)
                    self.logger.info(evaluation_prompt)
                    self.logger.info("="*80)

                user_messages.append(evaluation_prompt)
                valid_keys2eval.append(key)

        if not valid_keys2eval:
            return

        correct_syntax = False
        counter = 0
        while not correct_syntax and counter < 6:
            correct_syntax = True
            counter += 1

            responses = await asyncio.gather(
                *[self._async_llm_evaluate(msg) for msg in user_messages]
            )

            for i, response in enumerate(responses):
                key = valid_keys2eval[i]

                if self.logger:
                    self.logger.info(f"[HYP-EVAL-RESPONSE] {self.trader_id} for {key}:")
                    self.logger.info("-"*50)
                    self.logger.info(response)
                    self.logger.info("-"*50)

                is_correct = self._parse_evaluation_response(response)

                old_value = trader_hypotheses[key]['value']
                if is_correct:
                    prediction_error = self.correct_guess_reward - old_value
                else:
                    prediction_error = -self.correct_guess_reward - old_value

                trader_hypotheses[key]['value'] += self.alpha * prediction_error
                new_value = trader_hypotheses[key]['value']

                if new_value > self.good_hypothesis_thr:
                    good_hypothesis_found = True

                if self.logger:
                    self.logger.info(f"[HYP-EVAL-RW] {self.trader_id}: {key}")
                    self.logger.info(f"  Correct: {is_correct}, Value: {old_value:.3f} → {new_value:.3f}")
                    self.logger.info(f"  Rescorla-Wagner: {old_value:.3f} + {self.alpha} * {prediction_error:.3f}")

        if self.logger:
            self.logger.info(f"[HYP-EVAL-SUMMARY] {self.trader_id}: Evaluated {len(valid_keys2eval)} hypotheses for {opponent_id}")
            self.logger.info(f"  Good hypothesis found: {good_hypothesis_found} (threshold: {self.good_hypothesis_thr})")

    async def _async_llm_evaluate(self, prompt):
        """Helper method for async LLM evaluation"""
        response = self.model.generate_content(
            prompt,
            generation_config=self.model._generation_config
        )
        return response.text.strip()

    def get_good_hypotheses_data(self):
        """Return structured data of good hypotheses for JSON serialization"""
        good_hypotheses = {
            h_id: h_data for h_id, h_data in self.opponent_hypotheses.items()
            if h_data.get('value', 0) > self.good_hypothesis_thr
        }

        if not good_hypotheses:
            return []

        by_trader = {}
        for h_id, h_data in good_hypotheses.items():
            trader_id = h_data['target_trader']
            if trader_id not in by_trader:
                by_trader[trader_id] = []
            by_trader[trader_id].append(h_data)

        result = []
        for trader_id, hypotheses in by_trader.items():
            best_hypothesis = max(hypotheses, key=lambda h: h.get('value', 0))
            result.append({
                'opponent_id': str(trader_id),
                'hypothesis': best_hypothesis.get('possible_other_player_strategy', 'Unknown'),
                'value': best_hypothesis.get('value', 0.0),
                'count': len(hypotheses)
            })

        return result

    def get_good_hypotheses_context(self):
        """Return context string with all good hypotheses for decision making"""
        good_hypotheses = {
            h_id: h_data for h_id, h_data in self.opponent_hypotheses.items()
            if h_data.get('value', 0) > self.good_hypothesis_thr
        }

        if not good_hypotheses:
            return ""

        context = "\nOPPONENT STRATEGY INTELLIGENCE:\n"
        context += f"Based on {len(good_hypotheses)} proven hypotheses:\n\n"

        by_trader = {}
        for h_id, h_data in good_hypotheses.items():
            trader_id = h_data['target_trader']
            if trader_id not in by_trader:
                by_trader[trader_id] = []
            by_trader[trader_id].append(h_data)

        for trader_id, hypotheses in by_trader.items():
            best_hypothesis = max(hypotheses, key=lambda h: h.get('value', 0))

            context += f"Trader {trader_id} Strategy (HM value {best_hypothesis.get('value', 0):.2f}):\n"
            context += f"{best_hypothesis.get('possible_other_player_strategy', 'Unknown strategy')}\n\n"

        if self.logger:
            self.logger.info(f"[HYP-USE] {self.trader_id}: Including {len(good_hypotheses)} good hypotheses in decision context")

        return context

    def _get_agent_history(self, agent_id):
        """Get agent trading history from belief graph"""
        if not self.belief_graph or agent_id not in self.belief_graph.nodes:
            return {
                'total_trades': 0,
                'last_bid_price': None,
                'last_ask_price': None,
                'last_trade_price': None,
                'recent_events': []
            }

        agent_node = self.belief_graph.nodes[agent_id]
        recent_events = [
            event.to_dict() for event in self.belief_graph.event_history[-10:]
            if hasattr(event, 'agent_id') and event.agent_id == agent_id
        ]

        return {
            'total_trades': agent_node.total_trades,
            'last_bid_price': agent_node.last_bid_price,
            'last_ask_price': agent_node.last_ask_price,
            'last_trade_price': agent_node.last_trade_price,
            'recent_events': recent_events
        }

    def _create_hypothesis_prompt(self, opponent_id, action_data, market_context):
        """Create prompt for generating opponent strategy hypothesis"""
        agent_history = self._get_agent_history(opponent_id)

        event_type = action_data.get('event_type', 'trade')
        event_price = action_data.get('price', 'N/A')
        event_qty = action_data.get('quantity', 1)

        total_trades = agent_history.get('total_trades', 0)
        last_bid = agent_history.get('last_bid_price') or 'None'
        last_ask = agent_history.get('last_ask_price') or 'None'
        last_trade = agent_history.get('last_trade_price') or 'None'
        recent_events = agent_history.get('recent_events', [])

        recent_events_str = "\n".join([
            f"  - {e.get('event_type', 'unknown')} at price {e.get('price', 'N/A')} (qty: {e.get('quantity', 1)})"
            for e in recent_events[-5:]
        ]) if recent_events else "  No recent events"

        best_bid = market_context.get('best_bid') or 'N/A'
        best_ask = market_context.get('best_ask') or 'N/A'
        last_mkt_trade = market_context.get('last_trade_price') or 'N/A'
        recent_prices = market_context.get('recent_prices', [])
        spread = None
        if best_bid != 'N/A' and best_ask != 'N/A' and best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
        spread_str = spread if spread is not None else 'N/A'

        return f"""Analyze this trader's behavior and determine their strategy:

CURRENT EVENT:
Agent {opponent_id} just performed: {event_type} at price {event_price} (quantity: {event_qty})

AGENT'S TRADING HISTORY:
- Total trades completed: {total_trades}
- Last bid price: {last_bid}
- Last ask price: {last_ask}
- Last trade price: {last_trade}
- Recent activity:
{recent_events_str}

CURRENT MARKET STATE:
- Best bid: {best_bid}
- Best ask: {best_ask}
- Last market trade price: {last_mkt_trade}
- Bid-ask spread: {spread_str}
- Recent prices: {recent_prices}

Based on this action in this market context, what is this trader's likely strategy?
Consider: Are they aggressive (quick execution) or conservative (better margins)?
Do they follow momentum or mean reversion? Are they market makers?

Respond with a concise strategy description (50-100 words):"""

    def _create_evaluation_prompt(self, strategy_hypothesis, actual_action, market_context):
        """Create prompt for evaluating hypothesis accuracy"""
        opponent_id = actual_action.get('opponent_id', 'Unknown')
        agent_history = self._get_agent_history(opponent_id) if opponent_id != 'Unknown' else {}

        event_type = actual_action.get('event_type', 'trade')
        event_price = actual_action.get('price', 'N/A')
        event_qty = actual_action.get('quantity', 1)

        total_trades = agent_history.get('total_trades', 0)
        last_bid = agent_history.get('last_bid_price') or 'None'
        last_ask = agent_history.get('last_ask_price') or 'None'
        last_trade = agent_history.get('last_trade_price') or 'None'
        recent_events = agent_history.get('recent_events', [])

        recent_events_str = "\n".join([
            f"  - {e.get('event_type', 'unknown')} at price {e.get('price', 'N/A')} (qty: {e.get('quantity', 1)})"
            for e in recent_events[-5:]
        ]) if recent_events else "  No recent events"

        best_bid = market_context.get('best_bid') or 'N/A'
        best_ask = market_context.get('best_ask') or 'N/A'
        last_mkt_trade = market_context.get('last_trade_price') or 'N/A'
        recent_prices = market_context.get('recent_prices', [])
        spread = None
        if best_bid != 'N/A' and best_ask != 'N/A' and best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
        spread_str = spread if spread is not None else 'N/A'

        return f"""Does this trader's action match our hypothesis about their strategy?

OUR HYPOTHESIS: {strategy_hypothesis}

CURRENT EVENT:
Agent {opponent_id} just performed: {event_type} at price {event_price} (quantity: {event_qty})

AGENT'S TRADING HISTORY:
- Total trades completed: {total_trades}
- Last bid price: {last_bid}
- Last ask price: {last_ask}
- Last trade price: {last_trade}
- Recent activity:
{recent_events_str}

CURRENT MARKET STATE:
- Best bid: {best_bid}
- Best ask: {best_ask}
- Last market trade price: {last_mkt_trade}
- Bid-ask spread: {spread_str}
- Recent prices: {recent_prices}

Does their actual behavior align with our hypothesis?
Answer with: "YES - consistent with strategy" OR "NO - does not match strategy"
"""

    def _parse_evaluation_response(self, response_text):
        """Parse LLM evaluation response into boolean"""
        positive_indicators = ['yes', 'consistent', 'matches', 'aligns', 'correct', 'accurate']
        negative_indicators = ['no', 'inconsistent', 'does not match', 'wrong', 'incorrect']

        response_lower = response_text.lower()

        has_positive = any(indicator in response_lower for indicator in positive_indicators)
        has_negative = any(indicator in response_lower for indicator in negative_indicators)

        if has_positive and not has_negative:
            return True
        elif has_negative and not has_positive:
            return False
        else:
            return False

    def _prune_hypotheses(self, trader_id):
        """Keep only the best hypotheses for a trader"""
        trader_hypotheses = [
            (h_id, h_data) for h_id, h_data in self.opponent_hypotheses.items()
            if h_data['target_trader'] == trader_id
        ]

        if len(trader_hypotheses) > self.max_hypotheses_per_trader:
            trader_hypotheses.sort(key=lambda x: x[1].get('value', 0), reverse=True)

            to_remove = trader_hypotheses[self.max_hypotheses_per_trader:]
            for h_id, _ in to_remove:
                del self.opponent_hypotheses[h_id]

            if self.logger:
                self.logger.info(f"[HYP-PRUNE] {self.trader_id}: Pruned {len(to_remove)} hypotheses for {trader_id}")
