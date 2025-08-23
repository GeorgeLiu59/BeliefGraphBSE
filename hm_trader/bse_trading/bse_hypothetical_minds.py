"""
BSE Hypothetical-Minds Agent

Direct mirror of pd_hypothetical_minds.py adapted for Bristol Stock Exchange.
Maintains exact same structure and core logic as original HM implementation.

Core HM Logic Preserved:
- DecentralizedAgent base class pattern
- two_level_plan main decision function  
- Hypothesis generation and evaluation
- Rescorla-Wagner learning updates
- Memory state management
- LLM interaction patterns
"""

import re
import abc
import asyncio
import logging
import numpy as np
from copy import deepcopy
from queue import Queue
from typing import List, Dict, Any, Tuple, Optional

# Import BSE-specific dependencies
from BSE import Trader, Order


class HMFlowLogger:
    """
    ULTRA-VERBOSE logging for every stage of the Hypothetical-Minds agent flow.
    Routes all logs to a single HM-specific file for forensic-level debugging.
    PHILOSOPHY: Log EVERYTHING - no detail is too small during development.
    """
    
    def __init__(self, trader_id: str):
        self.trader_id = trader_id
        self.logger = logging.getLogger('BSE.LLM_HM')
        self.session_counter = 0
        self.stage_counters = {}
        
    def log_stage(self, stage_id: str, stage_name: str, details: str = "", data: Any = None, extra_context: Dict[str, Any] = None):
        """
        Log a specific stage with MAXIMUM verbosity
        """
        # Count stage occurrences
        stage_key = f"{stage_id}_{stage_name}"
        self.stage_counters[stage_key] = self.stage_counters.get(stage_key, 0) + 1
        count = self.stage_counters[stage_key]
        
        # Full data logging (no truncation)
        data_str = ""
        if data is not None:
            data_str = f" | FULL_DATA: {data}"
        
        # Extra context
        context_str = ""
        if extra_context:
            context_str = f" | CONTEXT: {extra_context}"
            
        self.logger.info(f"[{self.trader_id}] STAGE_{count} ({stage_id}) {stage_name} | {details}{data_str}{context_str}")
    
    def log_decision(self, decision_type: str, decision: Any, reasoning: str = "", confidence: float = None, alternatives: List[Any] = None):
        """Log decisions with full context and alternatives considered"""
        conf_str = f" | CONFIDENCE: {confidence}" if confidence is not None else ""
        alt_str = f" | ALTERNATIVES: {alternatives}" if alternatives else ""
        self.logger.info(f"[{self.trader_id}] DECISION {decision_type} | CHOSEN: {decision} | REASONING: {reasoning}{conf_str}{alt_str}")
        
    def log_learning(self, learning_type: str, before_state: Any, after_state: Any, prediction_accuracy: float = None):
        """Log learning with prediction accuracy"""
        acc_str = f" | ACCURACY: {prediction_accuracy:.3f}" if prediction_accuracy is not None else ""
        self.logger.info(f"[{self.trader_id}] LEARNING {learning_type} | BEFORE: {before_state} | AFTER: {after_state}{acc_str}")
        
    def log_error(self, stage: str, error: str, context: str = "", stack_trace: str = ""):
        """Log errors with full stack trace"""
        stack_str = f" | STACK: {stack_trace}" if stack_trace else ""
        self.logger.error(f"[{self.trader_id}] ERROR in {stage} | {error} | CONTEXT: {context}{stack_str}")
        
    def log_llm_interaction(self, message_type: str, prompt: str, response: str, tokens_used: int = None, latency_ms: float = None):
        """Log COMPLETE LLM interactions with performance metrics"""
        perf_str = ""
        if tokens_used or latency_ms:
            perf_str = f" | TOKENS: {tokens_used} | LATENCY: {latency_ms:.2f}ms"
        
        self.logger.info(f"[{self.trader_id}] LLM_{message_type} | FULL_PROMPT: {prompt} | FULL_RESPONSE: {response}{perf_str}")
        
    def log_market_state(self, time: float, lob: Dict[str, Any], balance: float, orders: List[Any]):
        """Log complete market state for decision context"""
        self.logger.info(f"[{self.trader_id}] MARKET_STATE | TIME: {time} | BALANCE: {balance} | ORDERS: {orders} | LOB: {lob}")
        
    def log_hypothesis_details(self, hypotheses: Dict[str, Any], good_found: bool, best_key: str = None):
        """Log detailed hypothesis state"""
        self.logger.info(f"[{self.trader_id}] HYPOTHESIS_STATE | COUNT: {len(hypotheses)} | GOOD_FOUND: {good_found} | BEST: {best_key} | ALL: {hypotheses}")
        
    def log_interaction_analysis(self, interaction_num: int, history: List[Dict], predictions: Dict[str, Any], actual_outcome: Dict[str, Any]):
        """Log complete interaction analysis"""
        self.logger.info(f"[{self.trader_id}] INTERACTION_{interaction_num} | HISTORY: {history} | PREDICTIONS: {predictions} | ACTUAL: {actual_outcome}")




class DecentralizedAgent(Trader, abc.ABC):
    """
    EXACT MIRROR: Hypothetical-Minds DecentralizedAgent for BSE trading
    
    Preserves HM structure:
    - Same initialization pattern
    - Same core attributes  
    - Same method signatures
    - Same hypothesis management
    """
    
    def __init__(
            self, 
            ttype: str,
            tid: str, 
            balance: float,
            params: Optional[Dict[str, Any]],
            time: float,
            config: Dict[str, Any],
            controller: Any,
            ) -> None:
        
        # (3) AGENT CREATE: Initialize authentic HM DecentralizedAgent for BSE trading
        # 
        # INTUITION: This is the core HM agent coming to life. We inherit from BSE's Trader
        # but add HM's sophisticated reasoning: opponent modeling, hypothesis generation,
        # Rescorla-Wagner learning, and strategic decision-making. The agent is now ready
        # to observe, hypothesize, predict, and adapt in the trading environment.
        
        # Initialize BSE Trader base class
        Trader.__init__(self, ttype, tid, balance, params, time)
        
        # Initialize HM Flow Logger
        self.hm_logger = HMFlowLogger(tid)
        self.hm_logger.log_stage("3", "AGENT_CREATE", f"Initializing DecentralizedAgent with config: {config}")
        
        # EXACT HM PATTERN: Core agent configuration
        self.agent_id = config['agent_id'] if 'agent_id' in config else tid
        self.config = config
        self.controller = controller        
        self.all_actions = Queue()
        self.generate_system_message()
        
        # EXACT HM PATTERN: Memory and interaction tracking
        self.memory_states = {}
        self.interact_steps = 0
        self.interaction_history = []
        self.opponent_hypotheses = {}
        self.interaction_num = 0
        self.good_hypothesis_found = False
        
        # EXACT HM PATTERN: Learning parameters
        self.alpha = 0.3  # learning rate for updating hypothesis values
        self.correct_guess_reward = 1
        self.good_hypothesis_thr = 0.7
        self.top_k = 5  # number of top hypotheses to evaluate
        self.self_improve = config.get('self_improve', True)
        
        # BSE-specific memory entities (adapted from HM entity types)
        # Original HM: ['green_box', 'red_box', 'ground', player_key, opponent_key]
        # BSE equivalent: market entities
        for entity_type in ['market_state', 'competitor_actions', 'price_levels']:
            self.memory_states[entity_type] = []
        
        # Initialize logging
        from logging_config import TraderLogger
        self.trader_logger = TraderLogger(tid, ttype)
        
        # BSE-specific state
        self.max_history = 50
        self.possible_opponent_strategy = None
        self.possible_opponent_price = None

    def generate_system_message(self):
        """
        EXACT MIRROR: HM's generate_system_message pattern
        Adapted for BSE trading context
        """
        self.system_message = f"""
            You are Agent {self.agent_id} in the Bristol Stock Exchange (BSE) trading simulation.
            This is a multi-agent financial trading environment where you compete with other traders 
            to maximize profit through strategic order placement.
            
            Your goal is to maximize trading profit over time.
            
            You can submit BID orders (to buy) or ASK orders (to sell) at specific prices.
            The market operates as a continuous double auction with a limit order book (LOB).
            Orders are matched when bid prices meet or exceed ask prices.
            
            You will repeatedly interact with other traders through order submissions and trades.
            This creates a strategic environment where you must model opponent behavior patterns
            to optimize your long-term trading performance.
            
            Key strategic considerations:
            - Track competitor trading patterns and adapt accordingly
            - Balance aggressive vs conservative pricing strategies  
            - Consider market momentum and liquidity conditions
            - Model opponent strategies to predict their future actions
            - Use hypothesis-driven reasoning to improve decision making
            
            Your responses should demonstrate strategic thinking about opponent modeling
            and long-term profit maximization in this repeated trading environment.
            """

    def update_memory_states(self, lob: Dict[str, Any], time: float):
        """
        EXACT MIRROR: HM's memory update pattern
        Adapted for BSE market observations
        """
        # Market state observation (equivalent to HM entity observations)
        market_observation = {
            'time': time,
            'best_bid': lob.get('bids', {}).get('best'),
            'best_ask': lob.get('asks', {}).get('best'),
            'spread': None
        }
        
        if market_observation['best_bid'] and market_observation['best_ask']:
            market_observation['spread'] = market_observation['best_ask'] - market_observation['best_bid']
        
        # EXACT HM PATTERN: Update memory with tuple format (observation, step_info, distance_info)
        entity_type = 'market_state'
        if entity_type not in self.memory_states:
            self.memory_states[entity_type] = []
        
        # HM pattern: Remove older references of same time and add new one
        self.memory_states[entity_type] = [
            obs for obs in self.memory_states[entity_type] 
            if not (isinstance(obs, tuple) and len(obs) >= 3 and 
                   isinstance(obs[0], dict) and obs[0].get('time') == time)
        ]
        self.memory_states[entity_type].append((
            market_observation, 
            f'Step: {int(time)}', 
            'distance: current'
        ))
        
        # Track competitor actions from tape
        if lob.get('tape') and len(lob['tape']) > 0:
            recent_trades = lob['tape'][-5:]
            for trade in recent_trades:
                if trade.get('type') == 'Trade' and trade.get('price'):
                    competitor_action = {
                        'time': trade.get('time', time),
                        'price': trade['price'],
                        'party1': trade.get('party1'),
                        'party2': trade.get('party2')
                    }
                    
                    if 'competitor_actions' not in self.memory_states:
                        self.memory_states['competitor_actions'] = []
                    
                    self.memory_states['competitor_actions'].append((
                        competitor_action,
                        f'Trade at {competitor_action["time"]:.2f}',
                        'distance: recent'
                    ))
        
        # EXACT HM PATTERN: Keep memory manageable
        for key in self.memory_states:
            if len(self.memory_states[key]) > self.max_history:
                self.memory_states[key] = self.memory_states[key][-self.max_history:]

    def generate_interaction_feedback_user_message1(self, step: float):
        """
        EXACT MIRROR: HM's generate_interaction_feedback_user_message1
        Adapted for BSE trading context - predicts opponent's next action
        """
        if not self.interaction_history:
            return "No interaction history available yet."
            
        user_message = f"""
            Step {step}: {self.interaction_history[-1]}
            You quoted: {self.interaction_history[-1].get('your_quote', 'N/A')}
            
            PREDICT: What price will competing traders quote next?
            ANSWER: "Competing traders will likely quote around [PRICE] with [CONFIDENCE]% confidence."
            
            Keep response short and direct.
            """
        return user_message

    def generate_interaction_feedback_user_message2(self, reward_tracker: Any, step: float):
        """
        (4h) HYPOTHESIS BUILDING: Generate new opponent strategy hypotheses
        
        INTUITION: This is where the agent becomes a "market psychologist". Looking at
        opponent trading patterns, the agent forms theories: "Are they momentum traders?
        Mean reversion? Market makers?" These hypotheses drive future predictions and
        strategic decisions. It's the agent building a mental model of opponent minds.
        """
        recent_interactions = self.interaction_history[-3:] if len(self.interaction_history) >= 3 else self.interaction_history
        
        user_message = f"""
            Step {step}: Recent interactions: {recent_interactions}
            
            STRATEGY: What is the competitors' likely trading strategy?
            ANSWER: "They appear to be using [STRATEGY] strategy with [CONFIDENCE]% confidence."
            
            Keep response short.
            """
        return user_message

    def generate_interaction_feedback_user_message3(self, step: float, possible_opponent_strategy=None):
        """
        EXACT MIRROR: HM's generate_interaction_feedback_user_message3
        Adapted for BSE - predicts opponent next action based on strategy hypothesis
        """
        if possible_opponent_strategy is None:
            possible_opponent_strategy = self.possible_opponent_strategy or "general market participant"
            
        user_message = f"""
            Step {step}: Recent interactions: {self.interaction_history[-3:] if len(self.interaction_history) >= 3 else self.interaction_history}
            
            PREDICT: What price will the opponent quote next?
            ANSWER: "Competing traders will likely quote around [PRICE] with [CONFIDENCE]% confidence."
            
            Keep response short and direct.
            """
        return user_message

    def generate_interaction_feedback_user_message4(self, step: float, possible_opponent_strategy=None):
        """
        EXACT MIRROR: HM's generate_interaction_feedback_user_message4
        Adapted for BSE - generates my next strategy decision
        """
        if possible_opponent_strategy is None:
            possible_opponent_strategy = self.possible_opponent_strategy
            
        user_message = f"""
            Step {step}: Recent interactions: {self.interaction_history[-3:] if len(self.interaction_history) >= 3 else self.interaction_history}
            
            DECISION: What should your quote price be?
            ANSWER: "I should quote [PRICE] because [REASONING]."
            
            Keep response short and direct.
            """
        return user_message

    # ============================================================================
    # DESIGN CHOICE: BSE vs Original HM Architecture
    # ============================================================================
    # ORIGINAL HM: two_level_plan() does Learning + Action Planning together
    # BSE REQUIREMENT: getorder() needs action only, respond() needs learning only
    # 
    # SOLUTION: Split the monolithic two_level_plan into two separate methods:
    # 1. learn_from_interaction() - ONLY learning/hypothesis updates (for respond())
    # 2. plan_action_with_hypotheses() - ONLY action planning (for getorder())
    # 
    # This preserves HM's sophisticated reasoning while adapting to BSE's timing.
    # ============================================================================

    async def learn_from_interaction(self, lob: Dict[str, Any], time: float) -> None:
        """
        LEARNING PHASE: Extract and process opponent behavior from last interaction
        
        DESIGN CHOICE: This method contains ONLY the learning parts (lines 461-488) 
        from original HM two_level_plan(). Called from respond() after trade outcomes.
        
        INTUITION: "I just traded with someone. What can I learn about their strategy?
        How should I update my beliefs about how they behave?"
        """
        interaction_context = {
            'interaction_num': self.interaction_num,
            'total_history': len(self.interaction_history),
            'hypotheses_count': len(self.opponent_hypotheses),
            'good_hypothesis_found': self.good_hypothesis_found
        }
        self.hm_logger.log_stage("4g", "LEARNING_START", f"Beginning learning from interaction #{self.interaction_num}", 
                                extra_context=interaction_context)
        self.hm_logger.log_market_state(time, lob, self.balance, self.orders)
        
        if len(self.interaction_history) == 0:
            self.hm_logger.log_stage("4g", "LEARNING_SKIP", "No interaction history to learn from yet")
            return  # Nothing to learn from yet
            
        # Step 1: Analyze what opponent actually did (Message 1)
        self.hm_logger.log_stage("4g", "LEARNING_ANALYZE", "Generating Message 1: Analyze opponent behavior", 
                                data={'last_interaction': self.interaction_history[-1]})
        
        import time as time_module
        start_time = time_module.time()
        hls_user_msg1 = self.generate_interaction_feedback_user_message1(time) 
        
        responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg1])
        possible_opponent_price = responses[0]  # Already structured dict
        
        end_time = time_module.time()
        latency = (end_time - start_time) * 1000
        
        self.hm_logger.log_llm_interaction("Message1", hls_user_msg1, str(possible_opponent_price), latency_ms=latency)
        
        # Validate required key
        if 'predicted_other_trader_next_price' not in possible_opponent_price:
            self.hm_logger.log_error("4g", "Missing predicted_other_trader_next_price", str(possible_opponent_price))
            raise ValueError(f"CRITICAL ERROR: LLM failed to provide required 'predicted_other_trader_next_price' field. "
                           f"Response: {possible_opponent_price}. This indicates schema enforcement is broken or "
                           f"message content detection failed. FIX IMMEDIATELY - no fallbacks in development phase!")
            
        self.possible_opponent_price = possible_opponent_price
        self.hm_logger.log_decision("OPPONENT_ANALYSIS", possible_opponent_price.get('predicted_other_trader_next_price'), 
                                   possible_opponent_price.get('reasoning', 'No reasoning provided'))
        
        # Update interaction history with opponent analysis
        old_history = self.interaction_history[-1].copy()
        self.interaction_history[-1].update(self.possible_opponent_price)
        self.hm_logger.log_learning("INTERACTION_UPDATE", old_history, self.interaction_history[-1])
        
        # Step 2: Evaluate how good our previous predictions were
        if self.interaction_num > 1:
            self.hm_logger.log_stage("4h", "HYPOTHESIS_EVAL", f"Evaluating {len(self.opponent_hypotheses)} hypotheses")
            old_hypotheses = deepcopy(self.opponent_hypotheses)
            self.evaluate_opponent_hypotheses()
            self.hm_logger.log_learning("HYPOTHESIS_UPDATE", 
                                      f"{len(old_hypotheses)} hypotheses", 
                                      f"{len(self.opponent_hypotheses)} hypotheses, good_found: {self.good_hypothesis_found}")
        
        self.hm_logger.log_stage("4g", "LEARNING_COMPLETE", f"Learning completed for interaction #{self.interaction_num}")

    async def plan_action_with_hypotheses(self, lob: Dict[str, Any], time: float, countdown: float) -> Dict[str, Any]:
        """
        ACTION PLANNING PHASE: Use accumulated hypotheses to make strategic decisions
        
        DESIGN CHOICE: This method contains the action planning parts (lines 489-564 + initial)
        from original HM two_level_plan(). Called from getorder() when BSE asks for trades.
        
        INTUITION: "Based on everything I've learned about my opponents, what should I do now?
        If I have no history, use basic strategy. If I have opponent models, use strategic reasoning."
        """
        self.hm_logger.log_stage("4c", "STRATEGIC_PLANNING", f"Planning action with {len(self.opponent_hypotheses)} hypotheses, countdown: {countdown}")
        
        # Case 1: No trading history - use initial strategy
        if len(self.interaction_history) == 0:
            self.hm_logger.log_stage("4c", "INITIAL_STRATEGY", "No interaction history, using initial strategy")
            hls_user_msg = self.generate_hls_user_message(lob, time, countdown)
            self.hm_logger.log_llm_interaction("InitialStrategy", "Initial market strategy", hls_user_msg[:100])
            
            responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg])
            my_strategy = responses[0]  # Already structured dict
            
            self.hm_logger.log_decision("INITIAL_PRICE", my_strategy.get('my_next_quote_price'), 
                                      my_strategy.get('reasoning', 'No reasoning provided'))
            return my_strategy
            
        # Case 2: Have trading history - use full HM strategic reasoning
        self.hm_logger.log_stage("4c", "HYPOTHESIS_STRATEGY", f"Using strategic reasoning with {len(self.interaction_history)} past interactions")
        return await self._strategic_planning_with_hypotheses(lob, time, countdown)
    
    async def _strategic_planning_with_hypotheses(self, lob: Dict[str, Any], time: float, countdown: float) -> Dict[str, Any]:
        """
        Internal method: Full HM strategic reasoning using accumulated opponent models
        
        This implements the sophisticated HM reasoning from lines 489-564 of original two_level_plan
        
        Args:
            lob: Current limit order book (available for future market-aware strategy)
            time: Current simulation time
            countdown: Time remaining (available for urgency-based decisions)
        """
        strategy_context = {
            'lob_state': {'best_bid': lob.get('bids', {}).get('best', 'None'), 
                         'best_ask': lob.get('asks', {}).get('best', 'None')},
            'time_remaining': countdown,
            'trader_balance': self.balance,
            'pending_orders': len(self.orders),
            'good_hypothesis_found': self.good_hypothesis_found
        }
        
        # Generate new hypothesis if no good one exists
        if not self.good_hypothesis_found:
            self.hm_logger.log_stage("4c", "NEW_HYPOTHESIS_PATH", "Generating new opponent hypothesis - no good hypothesis found")
            
            # Step 1: Generate opponent strategy hypothesis (Message 2)
            self.hm_logger.log_stage("4c", "MESSAGE2", "Generating opponent strategy hypothesis", extra_context=strategy_context)
            
            import time as time_module
            start_time = time_module.time()
            hls_user_msg2 = self.generate_interaction_feedback_user_message2(None, time)
            
            responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg2])
            possible_opponent_strategy = responses[0]  # Already structured dict
            
            end_time = time_module.time()
            latency = (end_time - start_time) * 1000
            
            self.hm_logger.log_llm_interaction("Message2", hls_user_msg2, str(possible_opponent_strategy), latency_ms=latency)
            self.possible_opponent_strategy = possible_opponent_strategy
            
            # Store new hypothesis
            self.opponent_hypotheses[self.interaction_num] = deepcopy(possible_opponent_strategy)
            self.opponent_hypotheses[self.interaction_num]['value'] = 0  # Initialize value
            self.hm_logger.log_learning("NEW_HYPOTHESIS", 
                                      f"Interaction {self.interaction_num}", 
                                      possible_opponent_strategy.get('possible_other_player_strategy', 'Unknown strategy'))
            
            # Step 2: Predict opponent next action based on strategy (Message 3)
            self.hm_logger.log_stage("4c", "MESSAGE3", "Predicting opponent next action", data={'strategy_basis': possible_opponent_strategy})
            
            start_time = time_module.time()
            hls_user_msg3 = self.generate_interaction_feedback_user_message3(time, possible_opponent_strategy)
            responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg3])
            other_player_next_action = responses[0]  # Already structured dict
            end_time = time_module.time()
            latency = (end_time - start_time) * 1000
            
            self.hm_logger.log_llm_interaction("Message3", hls_user_msg3, str(other_player_next_action), latency_ms=latency)
            
            # Store prediction in hypothesis
            self.opponent_hypotheses[self.interaction_num]['other_player_next_action'] = other_player_next_action
            
            # Step 3: Generate my strategic response (Message 4)
            self.hm_logger.log_stage("4c", "MESSAGE4", "Generating my strategic response", 
                                   data={'opponent_strategy': possible_opponent_strategy, 'opponent_prediction': other_player_next_action})
            
            start_time = time_module.time()
            hls_user_msg4 = self.generate_interaction_feedback_user_message4(time, possible_opponent_strategy)
            responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg4])
            my_next_strategy = responses[0]  # Already structured dict
            end_time = time_module.time()
            latency = (end_time - start_time) * 1000
            
            self.hm_logger.log_llm_interaction("Message4", hls_user_msg4, str(my_next_strategy), latency_ms=latency)
            self.hm_logger.log_decision("NEW_HYPOTHESIS_DECISION", my_next_strategy.get('my_next_quote_price'), 
                                      my_next_strategy.get('reasoning', 'No reasoning provided'))
            
            return my_next_strategy
        
        else:
            # Use best existing hypothesis for strategic planning
            self.hm_logger.log_stage("4c", "BEST_HYPOTHESIS_PATH", f"Using best existing hypothesis from {len(self.opponent_hypotheses)} total")
            
            sorted_keys = sorted([key for key in self.opponent_hypotheses], 
                               key=lambda x: self.opponent_hypotheses[x]['value'], reverse=True)
            best_key = sorted_keys[0]
            
            # Ensure hypothesis meets threshold
            best_value = self.opponent_hypotheses[best_key]['value']
            assert best_value > self.good_hypothesis_thr, f"Best hypothesis value {best_value} below threshold {self.good_hypothesis_thr}"
            
            best_hypothesis = self.opponent_hypotheses[best_key]
            self.hm_logger.log_learning("BEST_HYPOTHESIS_SELECTED", 
                                      f"Key: {best_key}, Value: {best_value}", 
                                      best_hypothesis.get('possible_other_player_strategy', 'Unknown strategy'))
            
            # Predict opponent next action using best hypothesis
            self.hm_logger.log_stage("4c", "BEST_MESSAGE3", "Predicting opponent action with best hypothesis", 
                                   data={'best_hypothesis': best_hypothesis, 'hypothesis_value': best_value})
            
            import time as time_module
            start_time = time_module.time()
            hls_user_msg3 = self.generate_interaction_feedback_user_message3(time, best_hypothesis)
            responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg3])
            other_player_next_action = responses[0]  # Already structured dict
            end_time = time_module.time()
            latency = (end_time - start_time) * 1000
            
            self.hm_logger.log_llm_interaction("BestMessage3", hls_user_msg3, str(other_player_next_action), latency_ms=latency)
            
            # Update best hypothesis
            self.opponent_hypotheses[best_key]['other_player_next_action'] = other_player_next_action
            
            # Generate my strategic response based on best hypothesis
            self.hm_logger.log_stage("4c", "BEST_MESSAGE4", "Generating strategic response with best hypothesis", 
                                   data={'best_hypothesis': best_hypothesis, 'opponent_prediction': other_player_next_action})
            
            start_time = time_module.time()
            hls_user_msg4 = self.generate_interaction_feedback_user_message4(time, best_hypothesis)
            responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg4])
            my_next_strategy = responses[0]  # Already structured dict
            end_time = time_module.time()
            latency = (end_time - start_time) * 1000
            
            self.hm_logger.log_llm_interaction("BestMessage4", hls_user_msg4, str(my_next_strategy), latency_ms=latency)
            self.hm_logger.log_decision("BEST_HYPOTHESIS_DECISION", my_next_strategy.get('my_next_quote_price'), 
                                      my_next_strategy.get('reasoning', 'No reasoning provided'))
            
            return my_next_strategy

    def generate_hls_user_message(self, lob: Dict[str, Any], time: float, countdown: float):
        """
        EXACT MIRROR: HM's generate_hls_user_message pattern
        Generates initial high-level strategy message for BSE trading
        """
        market_info = f"Time: {time}, Countdown: {countdown}"
        market_info += f", Best Bid: {lob.get('bids', {}).get('best', 'None')}"
        market_info += f", Best Ask: {lob.get('asks', {}).get('best', 'None')}"
        
        if lob.get('tape') and len(lob['tape']) > 0:
            recent_trades = lob['tape'][-3:]
            market_info += f", Recent trades: {recent_trades}"
        
        user_message = f"""
            Current market state: {market_info}
            Your trading context: Balance: {self.balance}, Orders: {self.orders}
            
            This is your initial trading decision in the BSE market.
            Analyze the market conditions and provide your initial trading strategy.
            
            Consider:
            - Current market spread and liquidity
            - Your position and trading objectives  
            - Potential competitor strategies
            - Long-term profit maximization
            
            What should your initial quote price be? Provide your reasoning and confidence level.
            """
        return user_message

    def eval_hypotheses(self):
        """
        (4g) LEARNING: Evaluate and update opponent strategy hypotheses
        
        INTUITION: This is the agent's "learning engine". Using Rescorla-Wagner learning,
        the agent compares its predictions against reality. Good predictions increase
        hypothesis confidence, bad predictions decrease it. Over time, the agent builds
        increasingly accurate models of opponent behavior patterns.
        """
        eval_context = {
            'total_hypotheses': len(self.opponent_hypotheses),
            'good_hypothesis_found_before': self.good_hypothesis_found,
            'interaction_history_length': len(self.interaction_history),
            'alpha': self.alpha,
            'good_hypothesis_thr': self.good_hypothesis_thr,
            'correct_guess_reward': self.correct_guess_reward
        }
        self.hm_logger.log_stage("4h", "EVAL_HYPOTHESES_START", "Beginning hypothesis evaluation", extra_context=eval_context)
        
        if not self.opponent_hypotheses:
            self.hm_logger.log_stage("4h", "EVAL_HYPOTHESES_SKIP", "No hypotheses to evaluate")
            return
            
        # EXACT HM PATTERN: Get latest key and sort others
        latest_key = max(self.opponent_hypotheses.keys())
        sorted_keys = sorted([key for key in self.opponent_hypotheses if key != latest_key],
                           key=lambda x: self.opponent_hypotheses[x]['value'], 
                           reverse=True)
        keys2eval = sorted_keys[:self.top_k] + [latest_key]
        
        eval_setup = {
            'latest_key': latest_key,
            'sorted_keys': sorted_keys,
            'top_k': self.top_k,
            'keys2eval': keys2eval,
            'keys2eval_count': len(keys2eval)
        }
        self.hm_logger.log_stage("4h", "EVAL_SELECTION", f"Selected {len(keys2eval)} hypotheses for evaluation", data=eval_setup)
        
        # EXACT HM PATTERN: Reset good hypothesis flag
        self.good_hypothesis_found = False
        
        evaluation_results = []
        
        for key in keys2eval:
            self.hm_logger.log_stage("4h", "EVAL_HYPOTHESIS", f"Evaluating hypothesis {key}", 
                                   data={'hypothesis': self.opponent_hypotheses[key]})
            
            # EXACT HM PATTERN: Evaluate prediction accuracy
            if 'other_player_next_action' not in self.opponent_hypotheses[key]:
                self.hm_logger.log_stage("4h", "EVAL_SKIP", f"Hypothesis {key} missing other_player_next_action")
                continue
                
            # Get predicted vs actual trading behavior
            predicted_data = self.opponent_hypotheses[key]['other_player_next_action']
            
            if 'predicted_other_trader_next_price' not in predicted_data:
                error_context = {
                    'hypothesis_key': key,
                    'predicted_data': predicted_data,
                    'hypothesis': self.opponent_hypotheses[key]
                }
                self.hm_logger.log_error("4h", "Missing predicted_other_trader_next_price", error_context)
                raise ValueError(f"CRITICAL ERROR: Missing 'predicted_other_trader_next_price' in hypothesis evaluation. "
                               f"Predicted data: {predicted_data}. This should never happen with proper schema enforcement. "
                               f"FIX IMMEDIATELY - no silent fallbacks in development phase!")
                
            predicted_price = predicted_data['predicted_other_trader_next_price']
            
            # Get actual competitor behavior from recent interactions
            if self.interaction_history and len(self.interaction_history) > 0:
                last_interaction = self.interaction_history[-1]
                actual_price = last_interaction.get('actual_competitor_price')
                
                comparison_data = {
                    'hypothesis_key': key,
                    'predicted_price': predicted_price,
                    'actual_price': actual_price,
                    'last_interaction': last_interaction
                }
                
                if actual_price is not None:
                    # EXACT HM PATTERN: Binary evaluation of prediction accuracy
                    price_diff = abs(predicted_price - actual_price)
                    prediction_correct = price_diff < 10  # Trading threshold (similar to HM tie handling)
                    
                    old_value = self.opponent_hypotheses[key]['value']
                    
                    # EXACT HM RESCORLA-WAGNER UPDATE
                    if prediction_correct:
                        prediction_error = self.correct_guess_reward - old_value
                    else:
                        prediction_error = -self.correct_guess_reward - old_value
                    
                    # Update value using exact HM formula
                    new_value = old_value + (self.alpha * prediction_error)
                    self.opponent_hypotheses[key]['value'] = new_value
                    
                    # EXACT HM PATTERN: Check for good hypothesis
                    is_good_hypothesis = new_value > self.good_hypothesis_thr
                    if is_good_hypothesis:
                        self.good_hypothesis_found = True
                    
                    # Comprehensive evaluation result
                    eval_result = {
                        'hypothesis_key': key,
                        'predicted_price': predicted_price,
                        'actual_price': actual_price,
                        'price_diff': price_diff,
                        'prediction_correct': prediction_correct,
                        'old_value': old_value,
                        'prediction_error': prediction_error,
                        'alpha': self.alpha,
                        'new_value': new_value,
                        'is_good_hypothesis': is_good_hypothesis,
                        'good_hypothesis_threshold': self.good_hypothesis_thr
                    }
                    evaluation_results.append(eval_result)
                    
                    self.hm_logger.log_learning("HYPOTHESIS_UPDATE", 
                                              f"Key {key}: {old_value:.4f} → {new_value:.4f}", 
                                              eval_result)
                    
                else:
                    self.hm_logger.log_stage("4h", "EVAL_NO_ACTUAL_PRICE", f"Hypothesis {key} - no actual price available", 
                                           data=comparison_data)
            else:
                self.hm_logger.log_stage("4h", "EVAL_NO_HISTORY", f"Hypothesis {key} - no interaction history available")
        
        # Final evaluation summary
        final_summary = {
            'evaluations_completed': len(evaluation_results),
            'good_hypothesis_found_after': self.good_hypothesis_found,
            'hypothesis_values': {k: v['value'] for k, v in self.opponent_hypotheses.items()},
            'evaluation_results': evaluation_results
        }
        self.hm_logger.log_stage("4h", "EVAL_HYPOTHESES_COMPLETE", 
                               f"Hypothesis evaluation complete - good hypothesis found: {self.good_hypothesis_found}", 
                               extra_context=final_summary)

    # BSE Integration Methods
    
    def getorder(self, time: float, countdown: float, lob: Dict[str, Any]):
        """
        (4b) AGENT.ACT(): BSE calls this when agent needs to make a trading decision
        
        INTUITION: This is the main action interface. BSE asks "what do you want to trade?"
        The HM agent doesn't just react to current market - it activates the full HM
        reasoning pipeline: analyze situation, model opponents, generate hypotheses,
        and make strategic decisions using two_level_plan.
        """
        order_context = {
            'time': time,
            'countdown': countdown,
            'customer_orders': len(self.orders),
            'balance': self.balance,
            'interaction_history': len(self.interaction_history),
            'hypotheses_count': len(self.opponent_hypotheses),
            'good_hypothesis_found': self.good_hypothesis_found
        }
        self.hm_logger.log_stage("4b", "GET_ORDER_START", f"BSE requesting order at time {time:.2f}, countdown {countdown:.2f}", 
                                extra_context=order_context)
        self.hm_logger.log_market_state(time, lob, self.balance, self.orders)
        
        if len(self.orders) < 1:
            self.hm_logger.log_stage("4b", "NO_ORDERS", "No customer orders to process - returning None")
            return None
            
        if not self.controller:
            self.hm_logger.log_error("4b", "No LLM controller available", "Cannot generate strategic decision - returning None")
            return None
        
        try:
            # Log customer order details
            order = self.orders[0]
            customer_order_details = {
                'type': order.otype,
                'price_limit': order.price,
                'quantity': order.qty,
                'order_id': getattr(order, 'qid', 'Unknown')
            }
            self.hm_logger.log_stage("4b", "CUSTOMER_ORDER_DETAILS", "Processing customer order", data=customer_order_details)
            
            # DESIGN CHOICE: Use separated HM method for pure action planning
            # This method uses accumulated opponent hypotheses if available, or basic strategy if not
            # NO LEARNING happens here - that's reserved for respond() method
            
            import time as time_module
            decision_start = time_module.time()
            self.hm_logger.log_stage("4b", "STRATEGIC_DECISION_START", "Triggering HM strategic planning")
            
            decision_result = asyncio.run(
                self.plan_action_with_hypotheses(lob, time, countdown)
            )
            
            decision_end = time_module.time()
            decision_latency = (decision_end - decision_start) * 1000
            
            # Extract quote price from HM decision
            quote_price = decision_result.get('my_next_quote_price', 150)
            decision_reasoning = decision_result.get('reasoning', 'No reasoning provided')
            decision_confidence = decision_result.get('confidence', 'Unknown')
            
            decision_summary = {
                'raw_hm_price': quote_price,
                'reasoning': decision_reasoning,
                'confidence': decision_confidence,
                'decision_latency_ms': decision_latency
            }
            self.hm_logger.log_stage("4d", "HM_DECISION_COMPLETE", "HM strategic decision completed", data=decision_summary)
            
            # Create BSE order with price validation
            quoteprice = int(max(1, min(500, quote_price)))
            original_quoteprice = quoteprice
            
            # Ensure we don't violate customer limit price
            price_adjustment = False
            if order.otype == 'Bid' and quoteprice > order.price:
                self.hm_logger.log_decision("PRICE_LIMIT_BID", f"Bid {quoteprice} limited to customer max {order.price}")
                quoteprice = order.price
                price_adjustment = True
            elif order.otype == 'Ask' and quoteprice < order.price:
                self.hm_logger.log_decision("PRICE_LIMIT_ASK", f"Ask {quoteprice} limited to customer min {order.price}")
                quoteprice = order.price
                price_adjustment = True
            
            new_order = Order(self.tid, order.otype, quoteprice, order.qty, time, lob['QID'])
            self.lastquote = new_order
            
            final_order_summary = {
                'order_type': order.otype,
                'final_price': quoteprice,
                'quantity': order.qty,
                'hm_suggested_price': quote_price,
                'bse_adjusted_price': original_quoteprice,
                'customer_limit_applied': price_adjustment,
                'customer_limit_price': order.price
            }
            self.hm_logger.log_stage("4d", "ORDER_CREATION_COMPLETE", "BSE order created from HM decision", data=final_order_summary)
            self.hm_logger.log_decision("FINAL_ORDER", f"{order.otype} {quoteprice}", 
                                      f"HM: {quote_price} → BSE: {original_quoteprice} → Final: {quoteprice}")
            return new_order
            
        except Exception as e:
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'time': time,
                'trader_id': self.tid
            }
            self.hm_logger.log_error("4b", f"Order generation failed: {e}", error_details)
            print(f"HM order generation failed for {self.tid}: {e}")
            return None

    def respond(self, time: float, lob: Dict[str, Any], trade: Any, vrbs: bool):
        """
        (4f) EXECUTION: Process market events and update agent state
        
        INTUITION: This is where the agent "learns from experience". When a trade happens,
        the agent updates its memory, records the interaction, and prepares data for
        hypothesis evaluation. It's like the agent saying "I predicted X, actual was Y,
        let me update my beliefs about opponent strategies."
        """
        respond_context = {
            'time': time,
            'has_trade': bool(trade),
            'current_balance': self.balance,
            'n_trades': self.n_trades,
            'interaction_history_length': len(self.interaction_history),
            'hypotheses_count': len(self.opponent_hypotheses)
        }
        
        if trade:
            trade_details = {
                'price': getattr(trade, 'price', 'Unknown'),
                'qty': getattr(trade, 'qty', 'Unknown'),
                'buyer': getattr(trade, 'buyer', 'Unknown'),
                'seller': getattr(trade, 'seller', 'Unknown'),
                'timestamp': getattr(trade, 'timestamp', time)
            }
            respond_context['trade_details'] = trade_details
            self.hm_logger.log_stage("4f", "RESPOND_TRADE_START", f"Processing trade at time {time:.2f}", 
                                   extra_context=respond_context)
            self.hm_logger.log_market_state(time, lob, self.balance, self.orders)
        else:
            self.hm_logger.log_stage("4f", "RESPOND_NO_TRADE", f"No trade occurred at time {time:.2f}", 
                                   extra_context=respond_context)
            
        # Update profit tracking
        old_profit = self.profitpertime
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        profit_update = {
            'old_profitpertime': old_profit,
            'new_profitpertime': self.profitpertime,
            'balance': self.balance,
            'birthtime': self.birthtime
        }
        self.hm_logger.log_stage("4f", "PROFIT_UPDATE", "Updated profit per time metric", data=profit_update)
        
        # Update HM memory states
        memory_before = {
            'interact_steps': self.interact_steps,
            'interaction_num': self.interaction_num,
            'good_hypothesis_found': self.good_hypothesis_found
        }
        self.update_memory_states(lob, time)
        memory_after = {
            'interact_steps': self.interact_steps,
            'interaction_num': self.interaction_num,
            'good_hypothesis_found': self.good_hypothesis_found
        }
        self.hm_logger.log_learning("MEMORY_STATE_UPDATE", memory_before, memory_after)
        
        if trade:
            # Detailed interaction data construction
            trade_price = getattr(trade, 'price', None) if hasattr(trade, 'price') else trade.get('price') if isinstance(trade, dict) else None
            your_quote_price = self.lastquote.price if self.lastquote else None
            
            interaction_data = {
                'time': time,
                'event': 'trade',
                'trade_object': str(trade),
                'your_quote': your_quote_price,
                'actual_competitor_price': trade_price,
                'trade_involved_us': (getattr(trade, 'buyer', None) == self.tid or getattr(trade, 'seller', None) == self.tid),
                'price_difference': abs(trade_price - your_quote_price) if trade_price and your_quote_price else None
            }
            
            # Store interaction in history
            old_history_length = len(self.interaction_history)
            old_interact_steps = self.interact_steps
            old_interaction_num = self.interaction_num
            
            self.interaction_history.append(interaction_data)
            self.interact_steps += 1
            self.interaction_num += 1
            
            history_update = {
                'old_history_length': old_history_length,
                'new_history_length': len(self.interaction_history),
                'old_interact_steps': old_interact_steps,
                'new_interact_steps': self.interact_steps,
                'old_interaction_num': old_interaction_num,
                'new_interaction_num': self.interaction_num,
                'latest_interaction': interaction_data
            }
            self.hm_logger.log_stage("4f", "INTERACTION_RECORDED", f"Recorded interaction #{self.interaction_num}", 
                                   extra_context=history_update)
            
            # DESIGN CHOICE: Use separated HM method for pure learning
            # This method ONLY analyzes the trade outcome and updates opponent models
            # NO ACTION PLANNING happens here - that's reserved for getorder() method
            try:
                import time as time_module
                learning_start = time_module.time()
                self.hm_logger.log_stage("4f", "LEARNING_TRIGGER", "Triggering comprehensive HM learning from interaction")
                
                asyncio.run(self.learn_from_interaction(lob, time))
                
                learning_end = time_module.time()
                learning_latency = (learning_end - learning_start) * 1000
                
                learning_summary = {
                    'learning_latency_ms': learning_latency,
                    'interaction_analyzed': self.interaction_num,
                    'hypotheses_after_learning': len(self.opponent_hypotheses),
                    'good_hypothesis_found_after': self.good_hypothesis_found
                }
                self.hm_logger.log_stage("4f", "LEARNING_COMPLETE", "HM learning completed successfully", 
                                       data=learning_summary)
                
            except Exception as e:
                error_context = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'interaction_num': self.interaction_num,
                    'trader_id': self.tid,
                    'time': time
                }
                self.hm_logger.log_error("4f", f"Learning failed: {e}", error_context)
                print(f"HM learning failed in respond() for {self.tid}: {e}")
            
            # Log trade response for BSE compatibility
            if hasattr(self, 'trader_logger'):
                response_summary = "Updated HM interaction history and learned from outcome"
                self.trader_logger.log_trade_response(trade, response_summary)
                
            self.hm_logger.log_stage("4f", "RESPOND_TRADE_COMPLETE", 
                                   f"Trade response processing completed for interaction #{self.interaction_num}")
        else:
            self.hm_logger.log_stage("4f", "RESPOND_NO_TRADE_COMPLETE", "No-trade response processing completed")
        
        return None