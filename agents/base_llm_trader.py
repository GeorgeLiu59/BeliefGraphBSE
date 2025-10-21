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

        self.inventory = 0
        self.n_trades = 0
        self.purchase_prices = []

        # Proprietary trading state for buy/sell toggle
        self.job = 'Buy'  # flag switches between 'Buy' & 'Sell'
        self.last_purchase_price = None

        self.starting_balance = balance
        self.total_profit = 0.0
        self.trading_history = []
        self.max_history = 50

        self.last_belief_update_time = 0.0
        self.belief_update_interval = 10
        self.last_processed_tape_index = 0

        self.logger = self._setup_logger(ttype, tid)

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
        avg_purchase_price = sum(self.purchase_prices) / len(self.purchase_prices) if self.purchase_prices else None
        return {
            'balance': self.balance,
            'inventory': self.inventory,
            'avg_purchase_price': avg_purchase_price,
            'last_purchase_price': self.last_purchase_price,
            'n_trades': self.n_trades,
            'recent_prices': [],
            'job': self.job  # Add the buy/sell job state
        }

    def get_llm_decision(self, prompt: str, current_time: float = None) -> Dict[str, Any]:
        """Get decision from LLM"""
        if not self.model:
            return {'action': 'WAIT', 'price': None, 'reasoning': 'No model available'}

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

        decision = PromptParser.parse_trading_action(response.text.strip())

        self.logger.info(f'=== PARSED DECISION ===')
        self.logger.info(f'Action: {decision["action"]}')
        self.logger.info(f'Price: {decision.get("price", "N/A")}')
        self.logger.info(f'Reasoning: {decision.get("reasoning", "N/A")[:500]}')
        self.logger.info(f'Balance: ${self.balance:.2f}')
        self.logger.info(f'Inventory: {self.inventory}')

        return decision

    def getorder(self, time, countdown, lob, p_eq=None, q_eq=None, demand_curve=None, supply_curve=None):
        """Main trading method - to be implemented by subclasses"""
        self.logger.debug(f"[GETORDER] Called at time {time:.1f}")
        raise NotImplementedError("Subclasses must implement getorder()")

    def bookkeep(self, time, trade, order, verbose):
        """Update trader state after trade execution with buy/sell toggle pattern"""
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        self.n_trades += 1
        time_elapsed = time - self.birthtime
        self.profitpertime = self.balance / time_elapsed if time_elapsed > 0 else 0

        if trade['party1'] == self.tid or trade['party2'] == self.tid:
            transaction_price = trade['price']

            if trade['party1'] == self.tid:
                # Successfully BOUGHT a unit
                self.balance -= transaction_price
                self.last_purchase_price = transaction_price
                self.inventory = 1
                self.job = 'Sell'  # CRITICAL: Switch to selling mode

                self.trading_history.append({
                    'time': time,
                    'event': 'BOUGHT',
                    'price': transaction_price,
                    'new_balance': self.balance,
                    'new_job': self.job
                })

                self.logger.info(f'TRADE: BOUGHT at ${transaction_price:.2f} | Balance: ${self.balance:.2f} | Job: {self.job}')
                print(f"📦 {self.ttype} {self.tid} BOUGHT at ${transaction_price} | Balance: ${self.balance:.0f} | Job: {self.job}")

            elif trade['party2'] == self.tid:
                # Successfully SOLD a unit
                old_balance = self.balance
                self.balance += transaction_price

                if self.last_purchase_price is not None:
                    profit = transaction_price - self.last_purchase_price
                    emoji = "🟢" if profit >= 0 else "🔴"

                    self.trading_history.append({
                        'time': time,
                        'event': 'SOLD',
                        'price': transaction_price,
                        'profit': profit,
                        'new_balance': self.balance,
                        'new_job': self.job
                    })

                    self.logger.info(f'TRADE: SOLD at ${transaction_price:.2f} | Profit: ${profit:.2f} | Balance: ${self.balance:.2f}')
                    print(f"{emoji} {self.ttype} {self.tid} SOLD at ${transaction_price} | Profit: ${profit:.0f} | Balance: ${self.balance:.0f}")
                else:
                    profit = 0  # Fallback if purchase price missing
                    self.logger.warning(f'TRADE: SOLD without purchase price recorded')
                    print(f"🔴 {self.ttype} {self.tid} SOLD at ${transaction_price} | No purchase price | Balance: ${self.balance:.0f}")

                # Reset state and switch back to buying
                self.inventory = 0
                self.last_purchase_price = None
                self.job = 'Buy'  # CRITICAL: Switch back to buying mode

                # Update job in history after reset
                if len(self.trading_history) > 0:
                    self.trading_history[-1]['new_job'] = self.job

            if len(self.trading_history) > self.max_history:
                self.trading_history = self.trading_history[-self.max_history:]

    def respond(self, time, lob, trade, verbose):
        """Respond to market events - base implementation with comprehensive logging"""
        self.logger.debug(f"[RESPOND] Called at time {time:.1f}")

        # Subclasses can override this, but this provides default comprehensive logging
        pass

    def log_belief_graph_update(self, time, events_processed, lob=None):
        """Log belief graph state after updates - called by subclasses"""
        self.logger.info(f"[UPDATE-SUMMARY] Processed {events_processed} market events at time {time:.1f}")

        if hasattr(self, 'belief_graph'):
            self.logger.info(f"=== BELIEF GRAPH STATE AT TIME {time:.1f} ===")
            try:
                if hasattr(self.belief_graph, 'query_action') and lob:
                    current_market_state = {
                        'best_bid': lob.get('bids', {}).get('best'),
                        'best_ask': lob.get('asks', {}).get('best'),
                        'time_remaining': 0
                    }
                    structured_beliefs = self.belief_graph.query_action(self.tid, current_market_state)
                    self.logger.info(json.dumps(structured_beliefs, indent=2))
                else:
                    graph_json = self.belief_graph.to_json()
                    self.logger.info(graph_json)
                self.logger.info("="*80)
            except Exception as e:
                self.logger.error(f"[GRAPH-ERROR] Failed to serialize belief graph: {e}")

        # Log current agent list
        if hasattr(self, 'belief_graph') and hasattr(self.belief_graph, 'agents'):
            self.logger.info(f"[AGENTS-TRACKED] Currently tracking {len(self.belief_graph.agents)} agents: {self.belief_graph.agents}")

    def _is_prop_trader(self, trader_id: str) -> bool:
        """Check if trader_id belongs to a trader that should be tracked in belief graph"""
        # Track ALL traders (both proprietary P* and customer B*/S* traders)
        # This allows the perfect belief graph to get valuations from customer traders
        return trader_id.startswith('P') or trader_id.startswith('B') or trader_id.startswith('S')

    def process_and_log_market_events(self, time, lob, trade):
        """Process market events with comprehensive logging - only track prop traders"""
        from .belief_graph import MarketEvent, EventType

        if (time - self.last_belief_update_time) < self.belief_update_interval:
            return 0

        self.last_belief_update_time = time
        self.logger.info(f"[BELIEF-THROTTLE] {self.tid}: Running belief graph update at time {time:.1f}")

        if hasattr(self.belief_graph, 'asset_node'):
            self.logger.info(f"[LOB-ACCESS] {self.tid}: Retrieving market state from LOB at time {time:.1f}")

            if lob['bids']['n'] > 0:
                self.belief_graph.asset_node.current_best_bid = lob['bids']['best']
                self.logger.info(f"[LOB-DATA] best_bid from LOB: {lob['bids']['best']}")
            else:
                self.belief_graph.asset_node.current_best_bid = None
                self.logger.info(f"[LOB-DATA] No bids in LOB, cleared best_bid")

            if lob['asks']['n'] > 0:
                self.belief_graph.asset_node.current_best_ask = lob['asks']['best']
                self.logger.info(f"[LOB-DATA] best_ask from LOB: {lob['asks']['best']}")
            else:
                self.belief_graph.asset_node.current_best_ask = None
                self.logger.info(f"[LOB-DATA] No asks in LOB, cleared best_ask")

            if lob.get('tape') and len(lob['tape']) > 0:
                last_trade_event = None
                for i in range(len(lob['tape']) - 1, -1, -1):
                    if lob['tape'][i].get('type') == 'Trade':
                        last_trade_event = lob['tape'][i]
                        break
                if last_trade_event:
                    self.belief_graph.asset_node.last_trade_price = last_trade_event['price']
                    self.logger.info(f"[LOB-DATA] last_trade_price from LOB tape: {last_trade_event['price']} (tape length: {len(lob['tape'])})")

        events_processed = 0

        if lob.get('tape') and len(lob['tape']) > 0:
            tape_length = len(lob['tape'])
            new_entries_start = self.last_processed_tape_index
            self.logger.info(f"[TAPE-SCAN] Tape has {tape_length} entries, processing from index {new_entries_start}")

            for i in range(new_entries_start, tape_length):
                tape_entry = lob['tape'][i]
                if tape_entry.get('type') == 'Trade':
                    buyer_id = tape_entry['party1']
                    seller_id = tape_entry['party2']
                    trade_price = tape_entry['price']
                    trade_time = tape_entry['time']

                    buyer_is_prop = self._is_prop_trader(buyer_id)
                    seller_is_prop = self._is_prop_trader(seller_id)
                    self.logger.info(f"[TRADE-EVENT] Buyer: {buyer_id} (prop={buyer_is_prop}), Seller: {seller_id} (prop={seller_is_prop}), Price: ${trade_price}")

                    if buyer_id != self.tid and self._is_prop_trader(buyer_id):
                        event = MarketEvent(
                            event_id=f"trade_{buyer_id}_{trade_time}",
                            event_type=EventType.TRADE,
                            timestamp=trade_time,
                            agent_id=buyer_id,
                            price=trade_price,
                            quantity=1,
                            counterparty_id=seller_id
                        )
                        if buyer_id not in self.belief_graph.agents:
                            self.belief_graph.add_agent(buyer_id)
                            self.logger.info(f"[NEW-AGENT] Added prop trader {buyer_id} to belief graph")
                        self.belief_graph.update_beliefs(event)
                        events_processed += 1

                    if seller_id != self.tid and self._is_prop_trader(seller_id):
                        event = MarketEvent(
                            event_id=f"trade_{seller_id}_{trade_time}",
                            event_type=EventType.TRADE,
                            timestamp=trade_time,
                            agent_id=seller_id,
                            price=trade_price,
                            quantity=1,
                            counterparty_id=buyer_id
                        )
                        if seller_id not in self.belief_graph.agents:
                            self.belief_graph.add_agent(seller_id)
                            self.logger.info(f"[NEW-AGENT] Added prop trader {seller_id} to belief graph")
                        self.belief_graph.update_beliefs(event)
                        events_processed += 1

            self.last_processed_tape_index = tape_length

        if lob['bids']['n'] > 0:
            for bid in lob['bids']['lob']:
                if bid[0] != self.tid and self._is_prop_trader(bid[0]):
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
                        self.logger.info(f"[NEW-AGENT] Added prop trader {bid[0]} to belief graph")
                    self.belief_graph.update_beliefs(event)
                    events_processed += 1

        if lob['asks']['n'] > 0:
            for ask in lob['asks']['lob']:
                if ask[0] != self.tid and self._is_prop_trader(ask[0]):
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
                        self.logger.info(f"[NEW-AGENT] Added prop trader {ask[0]} to belief graph")
                    self.belief_graph.update_beliefs(event)
                    events_processed += 1

        return events_processed
