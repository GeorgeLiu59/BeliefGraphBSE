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
        
        # Use proper logging from logging_config.py instead of custom logger
        from logging_config import TraderLogger
        self.hm_logger = TraderLogger(tid, ttype)
        self.hm_logger.print_style_log('info', f"""=== HM AGENT CREATE ===
Trader: {tid}
Type: {ttype}
Config: {config}
Balance: {balance}
Time: {time}""")
        
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
        self.alpha = 0.6  # learning rate for updating hypothesis values (increased for faster learning)
        self.correct_guess_reward = 1
        self.good_hypothesis_thr = 0.3  # lowered threshold for more realistic hypothesis acceptance
        self.top_k = 5  # number of top hypotheses to evaluate
        self.self_improve = config.get('self_improve', True)
        
        # BSE-specific memory entities (adapted from HM entity types)
        # Original HM: ['green_box', 'red_box', 'ground', player_key, opponent_key]
        # BSE equivalent: market entities
        for entity_type in ['market_state', 'competitor_actions', 'price_levels']:
            self.memory_states[entity_type] = []
            
        self.hm_logger.print_style_log('debug', f"""=== MEMORY ENTITIES INITIALIZED ===
Trader: {self.tid}
Memory Entity Types: {list(self.memory_states.keys())}
This will track market observations for hypothesis generation""")
        
        # Note: hm_logger above already initializes proper logging - no need for separate trader_logger
        
        self.hm_logger.print_style_log('info', f"""=== DECENTRALIZED AGENT INITIALIZATION COMPLETE ===
Trader: {self.tid}
Agent ID: {self.agent_id}
Self Improve: {self.self_improve}
Learning Rate (Alpha): {self.alpha}
Good Hypothesis Threshold: {self.good_hypothesis_thr}
Top K Hypotheses: {self.top_k}
Controller Available: {self.controller is not None}
Agent is ready to begin Hypothetical-Minds reasoning!""")
        
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
            
        last_interaction = self.interaction_history[-1]
        user_message = f"""
            Market Analysis - Step {step}
            Last interaction: {last_interaction}
            Your quote: {last_interaction.get('your_quote', 'N/A')}
            Actual competitor price: {last_interaction.get('actual_competitor_price', 'N/A')}
            
            TASK: Based on this interaction, predict what price competing traders will quote next.
            
            Consider:
            - Their revealed pricing strategy
            - Market conditions and trends
            - Competition level and aggressiveness
            - Price range is 1-500
            
            ABSOLUTELY CRITICAL - SYSTEM WILL CRASH IF NOT FOLLOWED:
            - You MUST respond with ONLY the JSON format below
            - NO explanations, NO additional text, NO markdown formatting
            - ONLY the JSON block with ```json markers
            - Any other text will cause immediate system failure

            ```json
            {{
              "predicted_other_trader_next_price": 150,
              "confidence": 0.7,
              "reasoning": "Brief reasoning for this prediction"
            }}
            ```
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
        
        # Enhanced prompt with more specific trading strategy guidance
        user_message = f"""
            Trading Analysis - Step {step}
            Recent market interactions: {recent_interactions}
            
            TASK: Analyze the competitors' trading strategy based on their price quotes and behavior patterns.
            
            Consider these trading strategy types:
            - Aggressive: Quotes close to best bid/ask, seeks immediate execution
            - Conservative: Quotes with wider spreads, prioritizes profit margins
            - Momentum: Follows market trends, adjusts quotes based on price direction  
            - Mean Reversion: Counter-trend trading, exploits price swings
            - Market Making: Provides liquidity, maintains bid-ask spreads
            
            ABSOLUTELY CRITICAL - SYSTEM WILL CRASH IF NOT FOLLOWED:
            - You MUST respond with ONLY the JSON format below
            - NO explanations, NO additional text, NO markdown formatting
            - ONLY the JSON block with ```json markers
            - Any other text will cause immediate system failure

            ```json
            {{
              "possible_other_player_strategy": "Detailed analysis of their trading strategy and behavior pattern"
            }}
            ```
            """
        return user_message


    def generate_interaction_feedback_user_message3(self, step: float, possible_opponent_strategy=None):
        """
        EXACT MIRROR: HM's generate_interaction_feedback_user_message3
        Adapted for BSE - predicts opponent next action based on strategy hypothesis
        """
        if possible_opponent_strategy is None:
            possible_opponent_strategy = self.possible_opponent_strategy or "general market participant"
            
        strategy_text = possible_opponent_strategy.get('possible_other_player_strategy', 'unknown strategy') if isinstance(possible_opponent_strategy, dict) else str(possible_opponent_strategy)
            
        user_message = f"""
            Prediction Task - Step {step}
            Recent interactions: {self.interaction_history[-3:] if len(self.interaction_history) >= 3 else self.interaction_history}
            Opponent Strategy: {strategy_text}
            
            TASK: Predict the competitor's next price quote based on their identified strategy.
            
            Consider:
            - Their past pricing behavior and patterns
            - Current market conditions (best bid/ask)
            - Their strategic approach identified earlier
            - Typical price ranges in this market (1-500)
            
            ABSOLUTELY CRITICAL - SYSTEM WILL CRASH IF NOT FOLLOWED:
            - You MUST respond with ONLY the JSON format below
            - NO explanations, NO additional text, NO markdown formatting
            - ONLY the JSON block with ```json markers
            - Any other text will cause immediate system failure

            ```json
            {{
              "predicted_other_trader_next_price": 150,
              "confidence": 0.7
            }}
            ```
            """
        return user_message

    def generate_interaction_feedback_user_message4(self, step: float, possible_opponent_strategy=None):
        """
        EXACT MIRROR: HM's generate_interaction_feedback_user_message4
        Adapted for BSE - generates my next strategy decision
        """
        if possible_opponent_strategy is None:
            possible_opponent_strategy = self.possible_opponent_strategy
            
        strategy_text = possible_opponent_strategy.get('possible_other_player_strategy', 'unknown strategy') if isinstance(possible_opponent_strategy, dict) else str(possible_opponent_strategy)
        predicted_action = possible_opponent_strategy.get('other_player_next_action', {}) if isinstance(possible_opponent_strategy, dict) else {}
        predicted_price = predicted_action.get('predicted_other_trader_next_price', 'unknown') if predicted_action else 'unknown'
            
        user_message = f"""
            Strategic Decision - Step {step}
            Recent interactions: {self.interaction_history[-3:] if len(self.interaction_history) >= 3 else self.interaction_history}
            Opponent Strategy: {strategy_text}
            Predicted Opponent Next Price: {predicted_price}
            
            TASK: Decide your optimal quote price to maximize profit against this opponent.
            
            Strategic considerations:
            - If opponent is aggressive, consider defensive pricing
            - If opponent is conservative, consider aggressive pricing
            - Balance immediate execution vs. profit margins
            - Account for market momentum and liquidity
            - Price range must be 1-500
            
            ABSOLUTELY CRITICAL - SYSTEM WILL CRASH IF NOT FOLLOWED:
            - You MUST respond with ONLY the JSON format below
            - NO explanations, NO additional text, NO markdown formatting
            - ONLY the JSON block with ```json markers
            - Any other text will cause immediate system failure

            ```json
            {{
              "my_next_quote_price": 125,
              "reasoning": "Strategic reasoning for this price decision",
              "confidence": 0.8
            }}
            ```
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
        self.hm_logger.print_style_log('info', f"""=== LEARNING START (4g) ===
Trader: {self.tid}
Interaction #: {self.interaction_num}
Total History: {len(self.interaction_history)}
Hypotheses Count: {len(self.opponent_hypotheses)}
Good Hypothesis Found: {self.good_hypothesis_found}
Market Time: {time}
Balance: {self.balance}
Orders: {len(self.orders)}
LOB Best Bid: {lob.get('bids', {}).get('best', 'None')}
LOB Best Ask: {lob.get('asks', {}).get('best', 'None')}""")
        
        if len(self.interaction_history) == 0:
            self.hm_logger.print_style_log('info', f"""=== LEARNING SKIP (4g) ===
Trader: {self.tid}
Reason: No interaction history to learn from yet""")
            return  # Nothing to learn from yet
            
        # Step 1: Analyze what opponent actually did (Message 1)
        self.hm_logger.print_style_log('debug', f"""=== LEARNING ANALYZE (4g) ===
Trader: {self.tid}
Generating Message 1: Analyze opponent behavior
Last Interaction: {self.interaction_history[-1] if self.interaction_history else 'None'}""")
        
        import time as time_module
        start_time = time_module.time()
        hls_user_msg1 = self.generate_interaction_feedback_user_message1(time) 
        
        responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg1])
        raw_response = responses[0][0] if isinstance(responses[0], list) else responses[0]
        
        # Parse the LLM response properly
        possible_opponent_price = self.extract_dict(raw_response)
        
        self.hm_logger.print_style_log('debug', f"""=== LEARNING MESSAGE 1 PARSING ===
Trader: {self.tid}
Raw Response: {raw_response}
Parsed Price Data: {possible_opponent_price}""")
        
        end_time = time_module.time()
        latency = (end_time - start_time) * 1000
        
        self.hm_logger.print_style_log('debug', f"""=== LLM INTERACTION: Message1 ===
Trader: {self.tid}
Latency: {latency:.2f}ms
Prompt: {hls_user_msg1}
Response: {str(possible_opponent_price)}""")
        
        # Validate required key with improved error handling
        if 'predicted_other_trader_next_price' not in possible_opponent_price:
            self.hm_logger.print_style_log('error', f"""=== CRITICAL LLM ERROR (4g) ===
Trader: {self.tid}
Error: Missing predicted_other_trader_next_price
Response: {str(possible_opponent_price)}
Keys Found: {list(possible_opponent_price.keys()) if isinstance(possible_opponent_price, dict) else 'Not a dict'}
Response Type: {type(possible_opponent_price)}
This indicates schema enforcement is broken - FIX IMMEDIATELY!""")
            
            # Try to extract a reasonable fallback value to prevent total failure
            fallback_price = None
            if isinstance(possible_opponent_price, dict):
                # Look for alternative key patterns
                for key in possible_opponent_price.keys():
                    if 'price' in key.lower() and isinstance(possible_opponent_price[key], (int, float)):
                        fallback_price = possible_opponent_price[key]
                        break
            
            if fallback_price:
                self.hm_logger.print_style_log('warning', f"""=== USING FALLBACK PRICE ===
Trader: {self.tid}
Found alternative price: {fallback_price}
Using this to continue learning process""")
                possible_opponent_price['predicted_other_trader_next_price'] = fallback_price
            else:
                raise ValueError(f"CRITICAL ERROR: LLM failed to provide required 'predicted_other_trader_next_price' field. "
                               f"Response: {possible_opponent_price}. This indicates schema enforcement is broken or "
                               f"message content detection failed. FIX IMMEDIATELY - no fallbacks in development phase!")
            
        self.possible_opponent_price = possible_opponent_price
        self.hm_logger.print_style_log('info', f"""=== OPPONENT ANALYSIS DECISION ===
Trader: {self.tid}
Predicted Price: {possible_opponent_price.get('predicted_other_trader_next_price')}
Reasoning: {possible_opponent_price.get('reasoning', 'No reasoning provided')}""")
        
        # Update interaction history with opponent analysis
        old_history = self.interaction_history[-1].copy()
        self.interaction_history[-1].update(self.possible_opponent_price)
        self.hm_logger.print_style_log('debug', f"""=== INTERACTION UPDATE ===
Trader: {self.tid}
Before: {old_history}
After: {self.interaction_history[-1]}""")
        
        # Step 1.5: Create new hypothesis about this opponent's strategy
        await self._create_hypothesis_from_observation(time)
        
        # Step 2: Evaluate how good our previous predictions were
        if self.interaction_num > 1:
            self.hm_logger.print_style_log('info', f"""=== HYPOTHESIS EVAL START (4h) ===
Trader: {self.tid}
Evaluating {len(self.opponent_hypotheses)} hypotheses
Interaction: {self.interaction_num}""")
            old_hypotheses = deepcopy(self.opponent_hypotheses)
            old_good_found = self.good_hypothesis_found
            # Get the opponent trader ID from the latest interaction
            latest_interaction = self.interaction_history[-1]
            opponent_trader_id = latest_interaction.get('opponent_trader_id')
            
            if opponent_trader_id:
                await self.evaluate_opponent_hypotheses(opponent_trader_id)
            else:
                self.hm_logger.print_style_log('warning', f"""=== NO OPPONENT ID (4h) ===
Trader: {self.tid}
Cannot evaluate hypotheses without opponent trader ID
Skipping hypothesis evaluation for this interaction""")
            self.hm_logger.print_style_log('info', f"""=== HYPOTHESIS UPDATE ===
Trader: {self.tid}
Before: {len(old_hypotheses)} hypotheses, good_found: {old_good_found}
After: {len(self.opponent_hypotheses)} hypotheses, good_found: {self.good_hypothesis_found}
Hypothesis Values: {[(k, h.get('value', 0)) for k, h in self.opponent_hypotheses.items()]}""")
        
        self.hm_logger.print_style_log('info', f"""=== LEARNING COMPLETE (4g) ===
Trader: {self.tid}
Interaction #{self.interaction_num} learning completed
Final State - Hypotheses: {len(self.opponent_hypotheses)}, Good Found: {self.good_hypothesis_found}""")

    async def plan_action_with_hypotheses(self, lob: Dict[str, Any], time: float, countdown: float) -> Dict[str, Any]:
        """
        ACTION PLANNING PHASE: Use accumulated hypotheses to make strategic decisions
        
        DESIGN CHOICE: This method contains the action planning parts (lines 489-564 + initial)
        from original HM two_level_plan(). Called from getorder() when BSE asks for trades.
        
        INTUITION: "Based on everything I've learned about my opponents, what should I do now?
        If I have no history, use basic strategy. If I have opponent models, use strategic reasoning."
        """
        self.hm_logger.print_style_log('info', f"""=== STRATEGIC PLANNING (4c) ===
Trader: {self.tid}
Hypotheses Available: {len(self.opponent_hypotheses)}
Countdown: {countdown:.3f}
Interaction History: {len(self.interaction_history)}
Good Hypothesis Found: {self.good_hypothesis_found}""")
        
        # Case 1: No trading history - use initial strategy
        if len(self.interaction_history) == 0:
            self.hm_logger.print_style_log('info', f"""=== INITIAL STRATEGY (4c) ===
Trader: {self.tid}
Reason: No interaction history available
Using initial market analysis strategy""")
            hls_user_msg = self.generate_hls_user_message(lob, time, countdown)
            self.hm_logger.print_style_log('debug', f"""=== LLM INTERACTION: Initial Strategy ===
Trader: {self.tid}
Prompt Preview: {hls_user_msg[:200]}...""")
            
            responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg])
            raw_response = responses[0][0] if isinstance(responses[0], list) else responses[0]
            
            # Parse the LLM response properly
            my_strategy = self.extract_dict(raw_response)
            
            self.hm_logger.print_style_log('debug', f"""=== INITIAL STRATEGY PARSING ===
Trader: {self.tid}
Raw Response: {raw_response}
Parsed Strategy: {my_strategy}""")
            
            self.hm_logger.print_style_log('info', f"""=== INITIAL PRICE DECISION ===
Trader: {self.tid}
Price: {my_strategy.get('my_next_quote_price')}
Reasoning: {my_strategy.get('reasoning', 'No reasoning provided')}""")
            return my_strategy
            
        # Case 2: Have trading history - use full HM strategic reasoning
        self.hm_logger.print_style_log('info', f"""=== HYPOTHESIS STRATEGY (4c) ===
Trader: {self.tid}
Using strategic reasoning with {len(self.interaction_history)} past interactions
Available hypotheses: {len(self.opponent_hypotheses)}""")
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
            self.hm_logger.print_style_log('info', f"""=== NEW HYPOTHESIS PATH ===
Trader: {self.tid}
Reason: No good hypothesis found (threshold: {self.good_hypothesis_thr})
Will generate new opponent strategy hypothesis""")
            
            # Action planning should NOT create new hypotheses - that happens in learning phase
            # Instead, use a fallback strategy when no good hypotheses exist
            self.hm_logger.print_style_log('info', f"""=== NO GOOD HYPOTHESES - FALLBACK STRATEGY ===
Trader: {self.tid}
Available hypotheses: {len(self.opponent_hypotheses)}
Good hypothesis threshold: {self.good_hypothesis_thr}
Using basic market strategy since no reliable opponent models exist
LOB State: {strategy_context.get('lob_state', {})}""")
            
            # Use basic market strategy when no good hypotheses are available
            # This is similar to the initial strategy approach but for ongoing trading
            hls_user_msg = self.generate_hls_user_message(lob, time, countdown)
            self.hm_logger.print_style_log('debug', f"""=== LLM INTERACTION: Basic Strategy ===
Trader: {self.tid}
Using basic market analysis instead of hypothesis-based strategy
Prompt Preview: {hls_user_msg[:200]}...""")
            
            import time as time_module
            start_time = time_module.time()
            
            responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg])
            raw_response = responses[0][0] if isinstance(responses[0], list) else responses[0]
            
            # Parse the LLM response properly
            my_strategy = self.extract_dict(raw_response)
            end_time = time_module.time()
            latency = (end_time - start_time) * 1000
            
            self.hm_logger.print_style_log('debug', f"""=== BASIC STRATEGY PARSING ===
Trader: {self.tid}
Latency: {latency:.2f}ms
Raw Response: {raw_response}
Parsed Strategy: {my_strategy}""")
            
            self.hm_logger.print_style_log('info', f"""=== BASIC STRATEGY DECISION ===
Trader: {self.tid}
Price: {my_strategy.get('my_next_quote_price')}
Reasoning: {my_strategy.get('reasoning', 'No reasoning provided')}""")
            
            return my_strategy
        
        else:
            # Use best existing hypothesis for strategic planning
            self.hm_logger.print_style_log('info', f"""=== BEST HYPOTHESIS PATH ===
Trader: {self.tid}
Using best existing hypothesis from {len(self.opponent_hypotheses)} total
Good hypothesis threshold: {self.good_hypothesis_thr}""")
            
            sorted_keys = sorted([key for key in self.opponent_hypotheses], 
                               key=lambda x: self.opponent_hypotheses[x]['value'], reverse=True)
            best_key = sorted_keys[0]
            
            # Ensure hypothesis meets threshold
            best_value = self.opponent_hypotheses[best_key]['value']
            assert best_value > self.good_hypothesis_thr, f"Best hypothesis value {best_value} below threshold {self.good_hypothesis_thr}"
            
            best_hypothesis = self.opponent_hypotheses[best_key]
            self.hm_logger.print_style_log('info', f"""=== BEST HYPOTHESIS SELECTED ===
Trader: {self.tid}
Key: {best_key}
Value: {best_value:.4f}
Strategy: {best_hypothesis.get('possible_other_player_strategy', 'Unknown strategy')}
Meets Threshold: {best_value > self.good_hypothesis_thr} (>{self.good_hypothesis_thr})""")
            
            # Predict opponent next action using best hypothesis
            self.hm_logger.print_style_log('debug', f"""=== BEST MESSAGE 3: Predict With Best Hypothesis ===
Trader: {self.tid}
Using best hypothesis for prediction
Hypothesis Value: {best_value:.4f}
Strategy: {best_hypothesis.get('possible_other_player_strategy', 'Unknown')}""")
            
            import time as time_module
            start_time = time_module.time()
            hls_user_msg3 = self.generate_interaction_feedback_user_message3(time, best_hypothesis)
            responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg3])
            other_player_next_action = responses[0]  # Already structured dict
            end_time = time_module.time()
            latency = (end_time - start_time) * 1000
            
            self.hm_logger.print_style_log('debug', f"""=== LLM INTERACTION: BestMessage3 ===
Trader: {self.tid}
Latency: {latency:.2f}ms
Prompt: {hls_user_msg3}
Response: {str(other_player_next_action)}""")
            
            # Update best hypothesis
            self.opponent_hypotheses[best_key]['other_player_next_action'] = other_player_next_action
            
            # Generate my strategic response based on best hypothesis
            self.hm_logger.print_style_log('debug', f"""=== BEST MESSAGE 4: Strategic Response ===
Trader: {self.tid}
Generating strategic response with best hypothesis
Hypothesis Value: {best_value:.4f}
Opponent Prediction: {other_player_next_action.get('predicted_other_trader_next_price', 'Unknown')}""")
            
            start_time = time_module.time()
            hls_user_msg4 = self.generate_interaction_feedback_user_message4(time, best_hypothesis)
            responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg4])
            my_next_strategy = responses[0]  # Already structured dict
            end_time = time_module.time()
            latency = (end_time - start_time) * 1000
            
            self.hm_logger.print_style_log('debug', f"""=== LLM INTERACTION: BestMessage4 ===
Trader: {self.tid}
Latency: {latency:.2f}ms
Prompt: {hls_user_msg4}
Response: {str(my_next_strategy)}""")
            self.hm_logger.print_style_log('info', f"""=== BEST HYPOTHESIS DECISION ===
Trader: {self.tid}
Price: {my_next_strategy.get('my_next_quote_price')}
Reasoning: {my_next_strategy.get('reasoning', 'No reasoning provided')[:200]}...
Used Hypothesis: {best_key} (value: {best_value:.4f})""")
            
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
            Initial Trading Decision
            Market state: {market_info}
            Your context: Balance: {self.balance}, Customer Orders: {len(self.orders)}
            
            TASK: Make your initial trading decision for this BSE market session.
            
            Market Analysis:
            - Current spread: {lob.get('asks', {}).get('best', 500) - lob.get('bids', {}).get('best', 1) if lob.get('bids', {}).get('best') and lob.get('asks', {}).get('best') else 'Wide'}
            - Liquidity: {len(lob.get('bids', {}).get('lob', [])) + len(lob.get('asks', {}).get('lob', []))} orders in book
            - Time pressure: {countdown:.2f} remaining
            
            Strategic Options:
            - Aggressive: Quote close to best prices for quick execution
            - Conservative: Quote with margin for better profit
            - Market Making: Provide liquidity with bid-ask spread
            
            ABSOLUTELY CRITICAL - SYSTEM WILL CRASH IF NOT FOLLOWED:
            - You MUST respond with ONLY the JSON format below
            - NO explanations, NO additional text, NO markdown formatting
            - ONLY the JSON block with ```json markers
            - Any other text will cause immediate system failure

            ```json
            {{
              "my_next_quote_price": 150,
              "reasoning": "Initial market analysis and strategic approach",
              "confidence": 0.7
            }}
            ```
            """
        return user_message

    def evaluate_predicted_behavior(self, step: float, predicted_next_behavior: Dict[str, Any]) -> str:
        """
        Generate prompt to evaluate if prediction matched observed behavior
        
        EXACT MIRROR: HM's evaluate_predicted_behavior pattern adapted for trading
        """
        latest_interaction = self.interaction_history[-1] if self.interaction_history else {}
        actual_competitor_price = latest_interaction.get('actual_competitor_price', 'Unknown')
        
        user_message = f"""
            A trade occurred at step {step}.
            You previously predicted that the competitor would perform this trading behavior: {predicted_next_behavior}
            Here is the observed behavior of the competitor in this round: Price quoted: {actual_competitor_price}
            Did your prediction match the observed behavior?
            Concisely output True or False in the below Python dictionary format, parsable by `ast.literal_eval()` starting with ```python.
            Example response:
            ```python
            {{
              'evaluate_predicted_behavior': True
            }}
            ```
            """
        return user_message

    async def evaluate_opponent_hypotheses(self, target_trader_id: str):
        """
        (4g) LEARNING: Evaluate and update opponent strategy hypotheses
        
        EXACT MIRROR: HM's evaluation pattern using LLM-based True/False evaluation
        with asyncio.gather for batch processing and Rescorla-Wagner learning
        """
        # Filter hypotheses for the specific trader who just acted
        trader_hypotheses = {k: v for k, v in self.opponent_hypotheses.items() 
                           if v.get('target_trader_id') == target_trader_id}
        
        self.hm_logger.print_style_log('info', f"""=== EVAL HYPOTHESES START (4h) ===
Trader: {self.tid}
Target Trader: {target_trader_id}
Total Hypotheses: {len(self.opponent_hypotheses)}
Hypotheses for {target_trader_id}: {len(trader_hypotheses)}
Good Hypothesis Found Before: {self.good_hypothesis_found}
Interaction History Length: {len(self.interaction_history)}
Learning Rate (Alpha): {self.alpha}
Good Hypothesis Threshold: {self.good_hypothesis_thr}
Correct Guess Reward: {self.correct_guess_reward}""")
        
        if not trader_hypotheses:
            self.hm_logger.print_style_log('info', f"""=== EVAL HYPOTHESES SKIP (4h) ===
Trader: {self.tid}
Reason: No hypotheses to evaluate for target trader {target_trader_id}""")
            return
            
        # EXACT HM PATTERN: Get latest key and sort others (but only for this trader)
        latest_key = max(trader_hypotheses.keys())
        sorted_keys = sorted([key for key in trader_hypotheses if key != latest_key],
                           key=lambda x: trader_hypotheses[x]['value'], 
                           reverse=True)
        keys2eval = sorted_keys[:self.top_k] + [latest_key]
        
        # Only evaluate hypotheses that have predictions to compare
        valid_keys2eval = []
        user_messages = []
        
        for key in keys2eval:
            if 'other_player_next_action' in trader_hypotheses[key]:
                predicted_next_behavior = trader_hypotheses[key]['other_player_next_action']
                current_value = trader_hypotheses[key]['value']
                reasoning = trader_hypotheses[key].get('reasoning', 'No reasoning available')
                
                # Get current market info for context
                latest_interaction = self.interaction_history[-1] if self.interaction_history else {}
                actual_price = latest_interaction.get('actual_competitor_price', 'Unknown')
                
                self.hm_logger.print_style_log('info', f"""=== EVALUATING HYPOTHESIS (4h) ===
Trader: {self.tid}
Target Trader: {target_trader_id}
Hypothesis Key: {key}
Current Value: {current_value:.4f}
Predicted Behavior: {predicted_next_behavior}
Actual Behavior: Price {actual_price}
Original Reasoning: {reasoning}
Preparing for LLM evaluation against actual {target_trader_id} behavior""")
                
                current_step = len(self.interaction_history)  # Use interaction count as step
                hls_user_msg3 = self.evaluate_predicted_behavior(current_step, predicted_next_behavior)
                user_messages.append(hls_user_msg3)
                valid_keys2eval.append(key)
                
        if not valid_keys2eval:
            self.hm_logger.print_style_log('info', f"""=== EVAL HYPOTHESES SKIP (4h) ===
Trader: {self.tid}
Reason: No hypotheses with predictions to evaluate for {target_trader_id}""")
            return

        # EXACT HM PATTERN: Reset good hypothesis flag  
        self.good_hypothesis_found = False

        # EXACT HM PATTERN: Call LLM to ask if any of the predicted behaviors are correct
        # Make sure output dict syntax is correct with retry logic
        correct_syntax = False
        counter = 0
        while not correct_syntax and counter < 10:
            correct_syntax = True
            counter += 1
            
            # EXACT HM PATTERN: Gathering responses asynchronously
            try:
                responses = await asyncio.gather(
                    *[self.controller.async_batch_prompt(self.system_message, [user_msg])
                    for user_msg in user_messages]
                )
                
                for i in range(len(responses)):
                    key = valid_keys2eval[i]
                    response = responses[i][0] if isinstance(responses[i], list) else responses[i]
                    
                    # Log the evaluation prompt and raw response for debugging
                    self.hm_logger.print_style_log('debug', f"""=== LLM EVALUATION DETAILS (4h) ===
Trader: {self.tid}
Hypothesis Key: {key}
Evaluation Prompt: {user_messages[i]}
Raw LLM Response: {response}""")
                    
                    pred_label = self.extract_dict(response)
                    
                    # EXACT HM PATTERN: Check for required key and valid True/False value
                    if 'evaluate_predicted_behavior' not in pred_label or pred_label['evaluate_predicted_behavior'] not in [True, False]:
                        correct_syntax = False
                        self.hm_logger.print_style_log('warning', f"""=== PARSING ERROR (4h) ===
Trader: {self.tid}
Error parsing dictionary when extracting label, retrying...
Response: {response}
Parsed: {pred_label}
Expected key: evaluate_predicted_behavior with True/False value""")
                        break
                    
                    # EXACT HM PATTERN: Rescorla-Wagner update based on True/False evaluation
                    old_value = self.opponent_hypotheses[key]['value']
                    if pred_label['evaluate_predicted_behavior']:
                        # if true, increase value of hypothesis
                        prediction_error = self.correct_guess_reward - old_value
                    elif not pred_label['evaluate_predicted_behavior']:
                        prediction_error = -self.correct_guess_reward - old_value
                    else:
                        # something weird happened
                        prediction_error = 0
                        
                    # EXACT HM PATTERN: Apply Rescorla-Wagner learning update
                    self.opponent_hypotheses[key]['value'] += self.alpha * prediction_error
                    new_value = self.opponent_hypotheses[key]['value']
                    
                    self.hm_logger.print_style_log('info', f"""=== RESCORLA-WAGNER UPDATE (4h) ===
Trader: {self.tid}
Hypothesis: {key}
LLM Evaluation: {pred_label['evaluate_predicted_behavior']}
Old Value: {old_value:.4f}
Prediction Error: {prediction_error:.4f}
Alpha: {self.alpha}
New Value: {new_value:.4f}
Update: {old_value:.4f} + {self.alpha} * {prediction_error:.4f} = {new_value:.4f}""")

                    # EXACT HM PATTERN: Check if hypothesis is now good enough
                    if self.opponent_hypotheses[key]['value'] > self.good_hypothesis_thr:
                        self.good_hypothesis_found = True
                        self.hm_logger.print_style_log('info', f"""=== GOOD HYPOTHESIS FOUND! ===
Trader: {self.tid}
Hypothesis: {key}
Value: {new_value:.4f} > Threshold: {self.good_hypothesis_thr}
This hypothesis will now be used for strategic decision making!""")
                        
            except Exception as e:
                self.hm_logger.print_style_log('error', f"""=== LLM EVALUATION ERROR (4h) ===
Trader: {self.tid}
Counter: {counter}
Error: {str(e)}
Retrying evaluation...""")
                correct_syntax = False

        # Final summary
        hypothesis_summary = [(k, v['value']) for k, v in self.opponent_hypotheses.items()]
        self.hm_logger.print_style_log('info', f"""=== EVAL HYPOTHESES COMPLETE (4h) ===
Trader: {self.tid}
Evaluations: {len(valid_keys2eval)}
Good Hypothesis Found: {self.good_hypothesis_found}
Final Values: {hypothesis_summary}
Using EXACT HM evaluation pattern with LLM True/False evaluation!""")

    # BSE Integration Methods
    
    def getorder(self, time: float, countdown: float, lob: Dict[str, Any]):
        """
        (4b) AGENT.ACT(): BSE calls this when agent needs to make a trading decision
        
        INTUITION: This is the main action interface. BSE asks "what do you want to trade?"
        The HM agent doesn't just react to current market - it activates the full HM
        reasoning pipeline: analyze situation, model opponents, generate hypotheses,
        and make strategic decisions using two_level_plan.
        """
        self.hm_logger.print_style_log('info', f"""=== GET ORDER START (4b) ===
Trader: {self.tid}
Time: {time:.2f}, Countdown: {countdown:.2f}
Customer Orders: {len(self.orders)}
Balance: {self.balance}
Interaction History: {len(self.interaction_history)}
Hypotheses Count: {len(self.opponent_hypotheses)}
Good Hypothesis Found: {self.good_hypothesis_found}
LOB Best Bid: {lob.get('bids', {}).get('best', 'None')}
LOB Best Ask: {lob.get('asks', {}).get('best', 'None')}
Tape Length: {len(lob.get('tape', []))}""")
        
        if len(self.orders) < 1:
            self.hm_logger.print_style_log('info', f"""=== NO ORDERS (4b) ===
Trader: {self.tid}
Reason: No customer orders to process - returning None
This is normal - trader waits for customer orders""")
            return None
            
        if not self.controller:
            self.hm_logger.print_style_log('error', f"""=== NO LLM CONTROLLER (4b) ===
Trader: {self.tid}
Error: No LLM controller available
Cannot generate strategic decision - returning None""")
            return None
        
        try:
            # Log customer order details
            order = self.orders[0]
            self.hm_logger.print_style_log('info', f"""=== CUSTOMER ORDER DETAILS (4b) ===
Trader: {self.tid}
Order Type: {order.otype}
Price Limit: {order.price}
Quantity: {order.qty}
Order ID: {getattr(order, 'qid', 'Unknown')}""")
            
            # DESIGN CHOICE: Use separated HM method for pure action planning
            # This method uses accumulated opponent hypotheses if available, or basic strategy if not
            # NO LEARNING happens here - that's reserved for respond() method
            
            import time as time_module
            decision_start = time_module.time()
            self.hm_logger.print_style_log('info', f"""=== STRATEGIC DECISION START (4b) ===
Trader: {self.tid}
Triggering HM strategic planning
Available hypotheses: {len(self.opponent_hypotheses)}
Good hypothesis found: {self.good_hypothesis_found}""")
            
            decision_result = asyncio.run(
                self.plan_action_with_hypotheses(lob, time, countdown)
            )
            
            decision_end = time_module.time()
            decision_latency = (decision_end - decision_start) * 1000
            
            # Extract quote price from HM decision
            quote_price = decision_result.get('my_next_quote_price', 150)
            decision_reasoning = decision_result.get('reasoning', 'No reasoning provided')
            decision_confidence = decision_result.get('confidence', 'Unknown')
            
            self.hm_logger.print_style_log('info', f"""=== HM DECISION COMPLETE (4d) ===
Trader: {self.tid}
Raw HM Price: {quote_price}
Reasoning: {decision_reasoning[:200]}...
Confidence: {decision_confidence}
Decision Latency: {decision_latency:.2f}ms""")
            
            # Create BSE order with price validation
            quoteprice = int(max(1, min(500, quote_price)))
            original_quoteprice = quoteprice
            
            # Ensure we don't violate customer limit price
            price_adjustment = False
            if order.otype == 'Bid' and quoteprice > order.price:
                self.hm_logger.print_style_log('info', f"""=== PRICE LIMIT BID ===
Trader: {self.tid}
Bid {quoteprice} limited to customer max {order.price}""")
                quoteprice = order.price
                price_adjustment = True
            elif order.otype == 'Ask' and quoteprice < order.price:
                self.hm_logger.print_style_log('info', f"""=== PRICE LIMIT ASK ===
Trader: {self.tid}
Ask {quoteprice} limited to customer min {order.price}""")
                quoteprice = order.price
                price_adjustment = True
            
            new_order = Order(self.tid, order.otype, quoteprice, order.qty, time, lob['QID'])
            self.lastquote = new_order
            
            self.hm_logger.print_style_log('info', f"""=== ORDER CREATION COMPLETE (4d) ===
Trader: {self.tid}
Order Type: {order.otype}
Final Price: {quoteprice}
Quantity: {order.qty}
HM Suggested Price: {quote_price}
BSE Adjusted Price: {original_quoteprice}
Customer Limit Applied: {price_adjustment}
Customer Limit Price: {order.price}
=== FINAL ORDER ===
Decision Path: HM: {quote_price} → BSE: {original_quoteprice} → Final: {quoteprice}""")
            return new_order
            
        except Exception as e:
            self.hm_logger.print_style_log('error', f"""=== ORDER GENERATION FAILED (4b) ===
Trader: {self.tid}
Error Type: {type(e).__name__}
Error Message: {str(e)}
Time: {time}
This is a CRITICAL ERROR - HM agent cannot function without order generation!""")
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
        # Create response context for logging
        trade_details = {}
        if trade:
            trade_details = {
                'price': getattr(trade, 'price', 'Unknown'),
                'qty': getattr(trade, 'qty', 'Unknown'),
                'buyer': getattr(trade, 'buyer', 'Unknown'),
                'seller': getattr(trade, 'seller', 'Unknown'),
                'timestamp': getattr(trade, 'timestamp', time)
            }
        
        if trade:
            self.hm_logger.print_style_log('info', f"""=== RESPOND TRADE START (4f) ===
Trader: {self.tid}
Time: {time:.2f}
Current Balance: {self.balance}
Trades Completed: {self.n_trades}
Interaction History: {len(self.interaction_history)}
Hypotheses Count: {len(self.opponent_hypotheses)}
Trade Details:
  Price: {trade_details.get('price', 'Unknown')}
  Quantity: {trade_details.get('qty', 'Unknown')}
  Buyer: {trade_details.get('buyer', 'Unknown')}
  Seller: {trade_details.get('seller', 'Unknown')}
Market State:
  Best Bid: {lob.get('bids', {}).get('best', 'None')}
  Best Ask: {lob.get('asks', {}).get('best', 'None')}""")
        else:
            self.hm_logger.print_style_log('debug', f"""=== RESPOND NO TRADE (4f) ===
Trader: {self.tid}
Time: {time:.2f}
Current Balance: {self.balance}
Trades Completed: {self.n_trades}
Interaction History: {len(self.interaction_history)}
Hypotheses Count: {len(self.opponent_hypotheses)}""")
            
        # Update profit tracking
        old_profit = self.profitpertime
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        self.hm_logger.print_style_log('debug', f"""=== PROFIT UPDATE (4f) ===
Trader: {self.tid}
Old Profit Per Time: {old_profit}
New Profit Per Time: {self.profitpertime}
Balance: {self.balance}
Birth Time: {self.birthtime}""")
        
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
        self.hm_logger.print_style_log('debug', f"""=== MEMORY STATE UPDATE (4f) ===
Trader: {self.tid}
Before: {memory_before}
After: {memory_after}
Memory Entities: {list(self.memory_states.keys())}
Memory Sizes: {[(k, len(v)) for k, v in self.memory_states.items()]}""")
        
        if trade:
            # Detailed interaction data construction
            trade_price = getattr(trade, 'price', None) if hasattr(trade, 'price') else trade.get('price') if isinstance(trade, dict) else None
            
            # Get the opponent trader ID from this trade
            # BSE trade objects have party1 and party2 fields, not buyer/seller
            party1 = getattr(trade, 'party1', None) if hasattr(trade, 'party1') else trade.get('party1') if isinstance(trade, dict) else None
            party2 = getattr(trade, 'party2', None) if hasattr(trade, 'party2') else trade.get('party2') if isinstance(trade, dict) else None
            
            # Also check buyer/seller fields for compatibility
            trade_buyer = getattr(trade, 'buyer', None) if hasattr(trade, 'buyer') else trade.get('buyer') if isinstance(trade, dict) else None
            trade_seller = getattr(trade, 'seller', None) if hasattr(trade, 'seller') else trade.get('seller') if isinstance(trade, dict) else None
            
            # Identify which opponent trader we can evaluate
            opponent_trader_id = None
            # Check party1/party2 first (BSE format)
            if party1 and party1 != self.tid:
                opponent_trader_id = party1
            elif party2 and party2 != self.tid:
                opponent_trader_id = party2
            # Fallback to buyer/seller (alternative format)
            elif trade_buyer and trade_buyer != self.tid:
                opponent_trader_id = trade_buyer
            elif trade_seller and trade_seller != self.tid:
                opponent_trader_id = trade_seller
                
            interaction_data = {
                'time': time,
                'event': 'trade',
                'trade_object': str(trade),
                'actual_competitor_price': trade_price,
                'opponent_trader_id': opponent_trader_id
            }
            
            self.hm_logger.print_style_log('info', f"""=== INTERACTION DATA CONSTRUCTION ===
Trader: {self.tid}
Trade Price: {trade_price}
Opponent Trader ID: {opponent_trader_id}
This will be used to evaluate hypotheses specific to {opponent_trader_id}!""")
            
            # Store interaction in history
            old_history_length = len(self.interaction_history)
            old_interact_steps = self.interact_steps
            old_interaction_num = self.interaction_num
            
            self.interaction_history.append(interaction_data)
            self.interact_steps += 1
            self.interaction_num += 1
            
            self.hm_logger.print_style_log('info', f"""=== INTERACTION RECORDED (4f) ===
Trader: {self.tid}
Interaction Number: {old_interaction_num} → {self.interaction_num}
History Length: {old_history_length} → {len(self.interaction_history)}
Interact Steps: {old_interact_steps} → {self.interact_steps}
This interaction will trigger learning and hypothesis updates!
Latest Interaction: {interaction_data}""")
            
            # DESIGN CHOICE: Use separated HM method for pure learning
            # This method ONLY analyzes the trade outcome and updates opponent models
            # NO ACTION PLANNING happens here - that's reserved for getorder() method
            
            # Proceed with learning from this interaction
            
            try:
                import time as time_module
                learning_start = time_module.time()
                self.hm_logger.print_style_log('info', f"""=== LEARNING TRIGGER (4f) ===
Trader: {self.tid}
Triggering comprehensive HM learning from interaction
Will analyze opponent behavior and update hypothesis values
Current hypotheses: {len(self.opponent_hypotheses)}""")
                
                asyncio.run(self.learn_from_interaction(lob, time))
                
                learning_end = time_module.time()
                learning_latency = (learning_end - learning_start) * 1000
                
                self.hm_logger.print_style_log('info', f"""=== LEARNING COMPLETE (4f) ===
Trader: {self.tid}
Learning Latency: {learning_latency:.2f}ms
Interaction Analyzed: {self.interaction_num}
Hypotheses After Learning: {len(self.opponent_hypotheses)}
Good Hypothesis Found: {self.good_hypothesis_found}
Hypothesis Values: {[(k, h.get('value', 0)) for k, h in self.opponent_hypotheses.items()]}
HM learning pipeline completed successfully!""")
                
            except Exception as e:
                self.hm_logger.print_style_log('error', f"""=== LEARNING FAILED (4f) ===
Trader: {self.tid}
Error Type: {type(e).__name__}
Error Message: {str(e)}
Interaction Number: {self.interaction_num}
Time: {time}
This is a CRITICAL ERROR - learning failure breaks the HM feedback loop!""")
                print(f"HM learning failed in respond() for {self.tid}: {e}")
            
            # CRITICAL: Call parent bookkeep to update balance after successful trade
            if hasattr(trade, 'price') and len(self.orders) > 0:
                # Create trade dict in format expected by BSE bookkeep
                trade_dict = {
                    'price': trade.price,
                    'party1': getattr(trade, 'party1', None),
                    'party2': getattr(trade, 'party2', None),
                    'time': time,
                    'qty': getattr(trade, 'qty', 1)
                }
                super().bookkeep(time, trade_dict, self.orders[0], vrbs)
                self.hm_logger.print_style_log('info', f"""=== BALANCE UPDATE (4f) ===
Trader: {self.tid}
Balance updated to: {self.balance}
Trades completed: {self.n_trades}""")

            # Log trade response for BSE compatibility
            response_summary = "Updated HM interaction history and learned from outcome"
            self.hm_logger.log_trade_response(trade, response_summary)
                
            self.hm_logger.print_style_log('info', f"""=== RESPOND TRADE COMPLETE (4f) ===
Trader: {self.tid}
Trade response processing completed for interaction #{self.interaction_num}
Agent is now ready for next trading decision with updated knowledge""")
        else:
            self.hm_logger.print_style_log('debug', f"""=== RESPOND NO TRADE COMPLETE (4f) ===
Trader: {self.tid}
No-trade response processing completed""")
        
        return None
    
    async def _create_hypothesis_from_observation(self, time: float):
        """
        Create new hypothesis based on observed opponent behavior during learning phase.
        
        This is called during respond() when we observe trades, NOT during getorder().
        We generate hypotheses about opponents based on their revealed behavior.
        """
        if not self.interaction_history:
            return
            
        latest_interaction = self.interaction_history[-1]
        opponent_trader_id = latest_interaction.get('opponent_trader_id')
        
        if not opponent_trader_id:
            self.hm_logger.print_style_log('debug', f"""=== HYPOTHESIS CREATION SKIP ===
Trader: {self.tid}
Reason: No specific opponent trader identified in latest interaction
Cannot create targeted hypothesis without opponent ID""")
            return
            
        self.hm_logger.print_style_log('info', f"""=== CREATING HYPOTHESIS FROM OBSERVATION ===
Trader: {self.tid}
Target Opponent: {opponent_trader_id}
Observed Behavior: {latest_interaction.get('actual_competitor_price')} price
Will generate strategy hypothesis based on this observation""")
        
        # Generate hypothesis about this opponent's strategy
        import time as time_module
        start_time = time_module.time()
        hls_user_msg2 = self.generate_interaction_feedback_user_message2(None, time)
        responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg2])
        raw_response = responses[0][0] if isinstance(responses[0], list) else responses[0]
        
        possible_opponent_strategy = self.extract_dict(raw_response)
        end_time = time_module.time()
        latency = (end_time - start_time) * 1000
        
        self.hm_logger.print_style_log('debug', f"""=== LLM INTERACTION: Observation Hypothesis ===
Trader: {self.tid}
Latency: {latency:.2f}ms
Target: {opponent_trader_id}
Response: {str(possible_opponent_strategy)}""")
        
        if not possible_opponent_strategy or 'possible_other_player_strategy' not in possible_opponent_strategy:
            self.hm_logger.print_style_log('warning', f"""=== HYPOTHESIS CREATION FAILED ===
Trader: {self.tid}
Target: {opponent_trader_id}
Reason: Invalid LLM response for strategy hypothesis
Response: {possible_opponent_strategy}""")
            return
            
        # Generate specific prediction for this strategy hypothesis  
        prediction_start = time_module.time()
        hls_user_msg3 = self.generate_interaction_feedback_user_message3(time, possible_opponent_strategy)
        prediction_responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg3])
        raw_prediction = prediction_responses[0][0] if isinstance(prediction_responses[0], list) else prediction_responses[0]
        
        predicted_action = self.extract_dict(raw_prediction)
        prediction_end = time_module.time()
        prediction_latency = (prediction_end - prediction_start) * 1000
        
        self.hm_logger.print_style_log('debug', f"""=== LLM INTERACTION: Prediction ===
Trader: {self.tid}
Latency: {prediction_latency:.2f}ms
Target: {opponent_trader_id}
Prediction Response: {str(predicted_action)}""")
        
        # Store new hypothesis with target trader ID and prediction
        self.opponent_hypotheses[self.interaction_num] = deepcopy(possible_opponent_strategy)
        self.opponent_hypotheses[self.interaction_num]['value'] = 0  # Initialize value
        self.opponent_hypotheses[self.interaction_num]['target_trader_id'] = opponent_trader_id
        
        # CRITICAL: Add the prediction for evaluation
        if predicted_action and ('predicted_other_trader_next_price' in predicted_action or 'confidence' in predicted_action):
            self.opponent_hypotheses[self.interaction_num]['other_player_next_action'] = predicted_action
            self.hm_logger.print_style_log('info', f"""=== PREDICTION ADDED ===
Trader: {self.tid}
Target: {opponent_trader_id}
Predicted Action: {predicted_action}
This hypothesis can now be evaluated in future interactions!""")
        else:
            self.hm_logger.print_style_log('warning', f"""=== PREDICTION MISSING ===
Trader: {self.tid}
Target: {opponent_trader_id}
Prediction Response: {predicted_action}
Hypothesis will not be evaluatable without prediction!""")
        
        # Validate hypothesis has required fields before storing
        stored_hypothesis = self.opponent_hypotheses[self.interaction_num]
        has_prediction = 'other_player_next_action' in stored_hypothesis
        
        self.hm_logger.print_style_log('info', f"""=== OBSERVATION HYPOTHESIS STORED ===
Trader: {self.tid}
Hypothesis Key: {self.interaction_num}
Target Trader: {opponent_trader_id}
Strategy: {possible_opponent_strategy.get('possible_other_player_strategy', 'Unknown strategy')[:100]}...
Initial Value: 0
Has Prediction: {has_prediction}
Total Hypotheses: {len(self.opponent_hypotheses)}
Status: {'Ready for evaluation' if has_prediction else 'Missing prediction - will be skipped in evaluation'}""")
    
    def extract_dict(self, response):
        """
        Extract JSON dictionary from LLM response text
        
        This method handles various LLM response formats including:
        - Plain JSON objects
        - JSON wrapped in ```json blocks  
        - JSON wrapped in ```python blocks
        - Mixed text with JSON content
        """
        import json
        import re
        
        try:
            # If response is already a dictionary, return it directly
            if isinstance(response, dict):
                return response
                
            # First try to parse the response directly as JSON
            response_str = str(response)
            if response_str.strip().startswith('{') and response_str.strip().endswith('}'):
                return json.loads(response_str)
                
            # Look for JSON blocks with various markers - more robust patterns
            json_patterns = [
                r'```json\s*\n?(.*?)\n?\s*```',   # Standard JSON blocks
                r'```python\s*\n?(.*?)\n?\s*```', # Python dict blocks (convert to JSON)
                r'```\s*\n?({.*?})\s*\n?```',     # Generic code blocks with JSON
                r'({.*?})',                        # Any JSON-like object in text
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, response_str, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
                    try:
                        return json.loads(json_text)
                    except json.JSONDecodeError:
                        # Try to fix Python dict format to JSON
                        if 'python' in pattern:
                            try:
                                # Convert Python dict syntax to JSON
                                fixed_text = json_text.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null')
                                return json.loads(fixed_text)
                            except json.JSONDecodeError:
                                pass
                        continue
                        
            # If no JSON found, try to extract key-value pairs from text
            response_str = str(response)
            self.hm_logger.print_style_log('warning', f"""=== EXTRACT DICT FALLBACK ===
Trader: {self.tid}
Could not find JSON in response, using fallback parsing
Response: {response_str[:200]}...""")
            
            # No more fallbacks - if JSON parsing fails, raise clear error
            raise ValueError(f"Could not extract valid JSON from LLM response: {response_str[:300]}...")
            
        except Exception as e:
            response_str = str(response)
            self.hm_logger.print_style_log('error', f"""=== EXTRACT DICT ERROR ===
Trader: {self.tid}
Error: {str(e)}
Response: {response_str[:200]}...""")
            raise  # Re-raise the error instead of returning empty dict