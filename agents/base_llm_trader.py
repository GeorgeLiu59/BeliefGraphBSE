#!/usr/bin/env python3
"""
Base LLM Trader
Provides common functionality for all LLM trading agents.
NO FALLBACKS. DIRECT EXECUTION.
"""

import os
import sys
import google.generativeai as genai
from typing import Dict, Any, Optional
import json
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BSE import Trader
from .unified_prompts import PromptBuilder, PromptParser, BasePrompts


class BaseLLMTrader(Trader):
    """Base class for all LLM trading agents"""

    _opened_log_files = set()

    def __init__(self, ttype: str, tid: str, balance: float, params: Dict[str, Any], time: float):
        Trader.__init__(self, ttype, tid, balance, params, time)

        self.api_key = params.get('api_key') or os.getenv('GOOGLE_API_KEY')
        self.model_name = params.get('model_name', 'gemini-2.0-flash-lite')
        self.temperature = params.get('temperature', 0.3)
        self.max_tokens = params.get('max_tokens', 4000)

        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"Initialized {ttype} trader {tid} with model {self.model_name}")
        else:
            self.model = None
            print(f"Warning: No API key for {ttype} trader {tid}")

        self.job = 'Buy'
        self.last_purchase_price = None
        self.inventory = 0
        self.n_trades = 0

        self.starting_balance = balance
        self.total_profit = 0.0
        self.trading_history = []
        self.max_history = 50

        self.decision_cooldown = params.get('decision_cooldown', 0.5)
        self.last_decision_time = -999
        self.last_decision = None
        self.cooldown_hits = 0

        self.logger = self._setup_logger(ttype, tid)
        self.logger.info(f"[INIT] Decision cooldown: {self.decision_cooldown}s")

    def _setup_logger(self, ttype: str, tid: str) -> logging.Logger:
        """Set up dedicated logger for this agent variant"""
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)

        logger = logging.getLogger(f'{ttype}_{tid}')
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if logger.handlers:
            return logger

        log_file = logs_dir / f'{ttype}.log'

        if str(log_file) not in BaseLLMTrader._opened_log_files:
            mode = 'w'
            BaseLLMTrader._opened_log_files.add(str(log_file))
        else:
            mode = 'a'

        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        logger.info(f'=== Agent {tid} ({ttype}) initialized ===')
        return logger

    def extract_recent_prices(self, lob: Dict, n_prices: int = 5) -> list:
        """Extract recent transaction prices from LOB tape"""
        recent_prices = []
        tape_position = -1
        count = 0
        while count < n_prices and abs(tape_position) < len(lob['tape']):
            if lob['tape'][tape_position]['type'] == 'Trade':
                recent_prices.append(lob['tape'][tape_position]['price'])
                count += 1
            tape_position -= 1
        return recent_prices

    def build_trader_state(self) -> Dict[str, Any]:
        """Build current trader state dict"""
        return {
            'balance': self.balance,
            'job': self.job,
            'inventory': self.inventory,
            'last_purchase_price': self.last_purchase_price,
            'n_trades': self.n_trades,
            'recent_prices': []
        }

    def get_llm_decision(self, prompt: str, current_time: float = None) -> Dict[str, Any]:
        """Get decision from LLM with cooldown caching"""
        if not self.model:
            return {'action': 'WAIT', 'price': None, 'reasoning': 'No model available'}

        if current_time and self.last_decision and (current_time - self.last_decision_time) < self.decision_cooldown:
            self.cooldown_hits += 1
            self.logger.info(f"[COOLDOWN] Reusing decision from {self.last_decision_time:.1f}s (cooldown: {self.decision_cooldown}s, hits: {self.cooldown_hits})")
            return self.last_decision

        self.logger.info(f"=== LLM PROMPT AT TIME {current_time if current_time else 'N/A'} ===")
        self.logger.info("="*80)
        self.logger.info(prompt)
        self.logger.info("="*80)

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
        )

        self.logger.info(f"=== RAW LLM RESPONSE ===")
        self.logger.info(response.text.strip())
        self.logger.info("="*80)

        decision = PromptParser.parse_trading_action(
            response.text.strip(),
            self.job.upper()
        )

        self.logger.info(f'=== PARSED DECISION ===')
        self.logger.info(f'Action: {decision["action"]}')
        self.logger.info(f'Price: {decision.get("price", "N/A")}')
        self.logger.info(f'Reasoning: {decision.get("reasoning", "N/A")[:500]}')
        self.logger.info(f'Current Job: {self.job}')
        self.logger.info(f'Balance: ${self.balance:.2f}')
        self.logger.info(f'Inventory: {self.inventory}')

        if current_time:
            self.last_decision_time = current_time
            self.last_decision = decision

        return decision

    def getorder(self, time, countdown, lob, p_eq=None, q_eq=None, demand_curve=None, supply_curve=None):
        """Main trading method - to be implemented by subclasses"""
        self.logger.debug(f"[GETORDER] Called at time {time:.1f}")
        raise NotImplementedError("Subclasses must implement getorder()")

    def bookkeep(self, time, trade, order, verbose):
        """Update trader state after trade execution - proprietary traders don't use parent bookkeep"""
        # Proprietary traders don't have customer orders in self.orders[], so we skip parent Trader.bookkeep()
        # and handle everything ourselves

        # Add to blotter
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        # Update profit per time
        self.n_trades += 1
        time_elapsed = time - self.birthtime
        self.profitpertime = self.balance / time_elapsed if time_elapsed > 0 else 0

        if trade['party1'] == self.tid or trade['party2'] == self.tid:
            transaction_price = trade['price']

            if self.job == 'Buy':
                self.last_purchase_price = transaction_price
                self.inventory = 1
                self.job = 'Sell'
                self.last_decision = None
                self.logger.info(f"[COOLDOWN-RESET] Job changed to {self.job}, cooldown stats: {self.cooldown_hits} cache hits")
                self.cooldown_hits = 0
                self.trading_history.append({
                    'time': time,
                    'event': 'BOUGHT',
                    'price': transaction_price,
                    'new_job': self.job
                })
                self.logger.info(f'TRADE: BOUGHT at ${transaction_price:.2f} | Balance: ${self.balance:.2f} | Next job: {self.job}')
                print(f"💰 {self.ttype} {self.tid} BOUGHT at ${transaction_price} | Balance: ${self.balance:.0f}")
            elif self.job == 'Sell':
                profit = transaction_price - self.last_purchase_price
                self.total_profit += profit
                self.balance += profit
                self.inventory = 0
                self.job = 'Buy'
                self.last_decision = None
                self.logger.info(f"[COOLDOWN-RESET] Job changed to {self.job}, cooldown stats: {self.cooldown_hits} cache hits")
                self.cooldown_hits = 0
                self.trading_history.append({
                    'time': time,
                    'event': 'SOLD',
                    'price': transaction_price,
                    'profit': profit,
                    'new_job': self.job
                })
                self.logger.info(f'TRADE: SOLD at ${transaction_price:.2f} | Profit: ${profit:.2f} | Total Profit: ${self.total_profit:.2f} | Balance: ${self.balance:.2f}')
                if profit > 0:
                    print(f"🟢 {self.ttype} {self.tid} SOLD at ${transaction_price} | Profit: ${profit:.0f} | Balance: ${self.balance:.0f}")
                elif profit == 0:
                    print(f"🟡 {self.ttype} {self.tid} SOLD at ${transaction_price} | Break-even | Balance: ${self.balance:.0f}")
                else:
                    print(f"🔴 {self.ttype} {self.tid} SOLD at ${transaction_price} | Loss: ${abs(profit):.0f} | Balance: ${self.balance:.0f}")

            if len(self.trading_history) > self.max_history:
                self.trading_history = self.trading_history[-self.max_history:]

    def respond(self, time, lob, trade, verbose):
        """Respond to market events - base implementation with comprehensive logging"""
        self.logger.debug(f"[RESPOND] Called at time {time:.1f}")

        # Subclasses can override this, but this provides default comprehensive logging
        pass

    def log_belief_graph_update(self, time, events_processed):
        """Log belief graph state after updates - called by subclasses"""
        self.logger.info(f"[UPDATE-SUMMARY] Processed {events_processed} market events at time {time:.1f}")

        if hasattr(self, 'belief_graph'):
            self.logger.info(f"=== BELIEF GRAPH STATE AT TIME {time:.1f} ===")
            try:
                graph_json = self.belief_graph.to_json()
                self.logger.info(graph_json)
                self.logger.info("="*80)
            except Exception as e:
                self.logger.error(f"[GRAPH-ERROR] Failed to serialize belief graph: {e}")

        # Log current agent list
        if hasattr(self, 'belief_graph') and hasattr(self.belief_graph, 'agents'):
            self.logger.info(f"[AGENTS-TRACKED] Currently tracking {len(self.belief_graph.agents)} agents: {self.belief_graph.agents}")

    def process_and_log_market_events(self, time, lob, trade):
        """Process market events with comprehensive logging - shared by all belief graph traders"""
        from .belief_graph import MarketEvent, EventType

        events_processed = 0

        if trade and 'party1' in trade and 'party2' in trade:
            buyer_id = trade['party1']
            seller_id = trade['party2']
            trade_price = trade['price']

            self.logger.debug(f"[TRADE-EVENT] Buyer: {buyer_id}, Seller: {seller_id}, Price: ${trade_price}")

            if buyer_id != self.tid:
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
                    self.logger.info(f"[NEW-AGENT] Added buyer {buyer_id} to belief graph")
                self.belief_graph.update_beliefs(event)
                events_processed += 1

            if seller_id != self.tid:
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
                    self.logger.info(f"[NEW-AGENT] Added seller {seller_id} to belief graph")
                self.belief_graph.update_beliefs(event)
                events_processed += 1

        if lob['bids']['n'] > 0:
            for bid in lob['bids']['lob']:
                # bid = [tid, price, qty]
                if bid[0] != self.tid:
                    event = MarketEvent(
                        event_id=f"bid_{bid[0]}_{time}",
                        event_type=EventType.BID,
                        timestamp=time,
                        agent_id=bid[0],
                        price=bid[1],
                        quantity=bid[2]
                    )
                    if bid[0] not in self.belief_graph.agents:
                        self.belief_graph.add_agent(bid[0])
                    self.belief_graph.update_beliefs(event)
                    events_processed += 1

        if lob['asks']['n'] > 0:
            for ask in lob['asks']['lob']:
                # ask = [tid, price, qty]
                if ask[0] != self.tid:
                    event = MarketEvent(
                        event_id=f"ask_{ask[0]}_{time}",
                        event_type=EventType.ASK,
                        timestamp=time,
                        agent_id=ask[0],
                        price=ask[1],
                        quantity=ask[2]
                    )
                    if ask[0] not in self.belief_graph.agents:
                        self.belief_graph.add_agent(ask[0])
                    self.belief_graph.update_beliefs(event)
                    events_processed += 1

        return events_processed
