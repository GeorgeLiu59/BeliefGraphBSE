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
        
        # Initialize BSE Trader base class
        Trader.__init__(self, ttype, tid, balance, params, time)
        
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
            A trading interaction has occurred at step {step}, {self.interaction_history[-1]}.
            Total interaction history: {self.interaction_history}.
            You last quoted: {self.interaction_history[-1].get('your_quote', 'N/A')}
            
            Based on the trading pattern and market context, predict what the other traders will do next.
            
            Consider the BSE trading environment where:
            - Traders submit bid/ask orders with specific prices
            - Orders are matched when bid >= ask
            - Traders adapt strategies based on market conditions and competitor behavior
            - Profit maximization drives all decisions
            
            This response should include step-by-step reasoning:
            1. 'Other_traders_next_action': Given the trading history and market patterns, 
               what price level are competing traders likely to quote next?
            2. Output the prediction in Python dictionary format, parsable by ast.literal_eval()
            
            Example response:
            1. 'Other_traders_next_action': Based on competitor patterns, they likely will bid around 145-150 to stay competitive.
            ```python
            {{
              'predicted_other_trader_next_price': 147,
              'confidence': 0.7,
              'reasoning': 'Competitors have been consistently bidding within 5 cents of market price'
            }}
            ```
            """
        return user_message

    def generate_interaction_feedback_user_message2(self, reward_tracker: Any, step: float):
        """
        EXACT MIRROR: HM's generate_interaction_feedback_user_message2  
        Adapted for BSE - generates opponent strategy hypothesis
        """
        recent_interactions = self.interaction_history[-3:] if len(self.interaction_history) >= 3 else self.interaction_history
        
        user_message = f"""
            A trading interaction has occurred at step {step}. Recent interaction history: {recent_interactions}.
            Current reward/profit status: {reward_tracker if reward_tracker else 'Not available'}.
            
            What is the likely trading strategy of your competitors given their recent pricing behavior and market conditions?
            Think step by step about this given the trading history.
            
            They may be using:
            - Momentum-following strategies (buying when prices rise)  
            - Mean reversion strategies (buying when prices fall)
            - Spread-based strategies (maintaining fixed spreads)
            - Adaptive strategies (responding to your actions)
            - Market making strategies (providing liquidity)
            
            Consider:
            - Are they consistently pricing aggressively (close to best bid/ask)?
            - Do they respond to market changes or maintain consistent behavior?
            - Are they trying to undercut competitors or follow market trends?
            
            Format your response as a Python dictionary starting with ```python:
            
            Example response:
            ```python
            {{
                'possible_other_player_strategy': 'Competitors are following an aggressive momentum strategy, consistently pricing within 2-3 cents of the best market price to maximize trade execution probability.',
                'strategy_confidence': 0.8,
                'key_patterns': ['aggressive_pricing', 'momentum_following', 'quick_adaptation']
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
            
        user_message = f"""
            An interaction with other traders has occurred at step {step}, {self.interaction_history[-1] if self.interaction_history else {}}.
            Total interaction history: {self.interaction_history}.
            You last quoted: {self.interaction_history[-1].get('your_quote') if self.interaction_history else 'N/A'}
            You previously guessed that their strategy is: {possible_opponent_strategy}.
            
            Given this strategy and the interaction history, predict what the other traders will quote next.
            
            Remember the BSE trading dynamics:
            - Traders balance profit margins vs execution probability
            - Market conditions influence aggressive vs conservative pricing
            - Competitors adapt to each other's strategies over time
            
            Format response with step-by-step reasoning:
            1. 'Other_traders_next_price': Given the strategy guess and last interaction,
               what price are they likely to quote next?
            2. Output prediction in Python dictionary format
            
            Example response:
            1. 'Other_traders_next_price': Given their momentum strategy, they'll likely bid 148 to stay competitive.
            ```python
            {{
              'predicted_other_trader_next_price': 148,
              'confidence': 0.75
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
            
        user_message = f"""
            An interaction with other traders has occurred at step {step}, {self.interaction_history[-1] if self.interaction_history else {}}.
            Total interaction history: {self.interaction_history}.
            You last quoted: {self.interaction_history[-1].get('your_quote') if self.interaction_history else 'N/A'}
            You previously guessed their strategy is: {possible_opponent_strategy}.
            You predicted their next price will be: {self.possible_opponent_price}.
            
            High-level strategy request:            
            Given the above information, provide the next trading strategy to maximize long-term profit.
            
            Remember BSE trading principles:
            - This is a repeated game where reputation and adaptation matter
            - You want to balance immediate profit vs long-term strategic positioning
            - Consider how your actions will influence competitor behavior in future rounds
            
            Response format with step-by-step reasoning:
            1. My strategy: What should your long-term trading approach be? 
               If competitors are adaptive, think about actions that will lead them to quote prices better for you.
               This is a strategic environment - what pricing strategy will maximize long-term profits?
            2. My next quote: Given competitors' likely next prices and your strategy, what should you quote?
            3. Output your next quote in Python dictionary format
            
            Example response:
            1. My strategy: Given competitors use momentum strategies, I should quote slightly more aggressively to capture trades while they adjust.
            2. My next quote: To maximize profit I should quote 146 to undercut their likely 148 quote.
            ```python
            {{
              'my_next_quote_price': 146,
              'reasoning': 'Strategic undercutting to capture trades before competitors adapt'
            }}
            ```
            
            Your goal is to maximize profit over the entire trading session, so consider long-term consequences.
            """
        return user_message

    async def two_level_plan(self, lob: Dict[str, Any], time: float, countdown: float, after_interaction: bool = False):
        """
        EXACT MIRROR: HM's two_level_plan - the core decision-making function
        
        This is the heart of the Hypothetical-Minds approach, exactly mirroring the original
        structure and logic flow from pd_hypothetical_minds.py lines 457-636
        """
        if after_interaction:
            # EXACT HM PATTERN: After interaction processing
            hls_user_msg = ''
            hls_response = ''
            
            # Step 1: Predict opponent next action
            hls_user_msg1 = self.generate_interaction_feedback_user_message1(time)
            hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg1
            
            # EXACT HM PATTERN: Ensure correct syntax with retry logic
            correct_syntax = False
            counter = 0
            while not correct_syntax and counter < 6:
                correct_syntax = True
                responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg1])
                response = responses[0]                
                possible_opponent_price = self.extract_dict(response)
                
                # EXACT HM PATTERN: Validate key presence
                key_present = 'predicted_other_trader_next_price' in possible_opponent_price
                
                if not key_present:
                    correct_syntax = False
                    print(f"Error parsing dictionary when extracting other trader price, retrying...")
                counter += 1
                
            self.possible_opponent_price = possible_opponent_price
            print(f'Other trader price prediction: {possible_opponent_price}')
            
            # EXACT HM PATTERN: Add response and update interaction history
            hls_response = hls_response + '\n\n' + response
            self.interaction_history[-1].update(self.possible_opponent_price)
            
            # EXACT HM PATTERN: Evaluate hypotheses if we have enough history
            if self.interaction_num > 1:
                self.eval_hypotheses()
                
            # EXACT HM PATTERN: Generate new hypothesis if no good one found
            if not self.good_hypothesis_found:
                hls_user_msg2 = self.generate_interaction_feedback_user_message2(None, time)
                hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg2
                responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg2])
                response = responses[0]                
                possible_opponent_strategy = self.extract_dict(response)
                
                self.possible_opponent_strategy = possible_opponent_strategy
                print(f'Opponent strategy hypothesis: {possible_opponent_strategy}')
                hls_response = hls_response + '\n\n' + response
                
                # EXACT HM PATTERN: Store new hypothesis
                self.opponent_hypotheses[self.interaction_num] = deepcopy(possible_opponent_strategy)
                self.opponent_hypotheses[self.interaction_num]['value'] = 0  # Initialize value
                
                # Step 3: Predict opponent next action based on strategy
                hls_user_msg3 = self.generate_interaction_feedback_user_message3(time, possible_opponent_strategy)
                hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg3
                responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg3])
                response = responses[0]
                other_player_next_action = self.extract_dict(response)
                
                # EXACT HM PATTERN: Store prediction in hypothesis
                self.opponent_hypotheses[self.interaction_num]['other_player_next_action'] = other_player_next_action
                hls_response = hls_response + '\n\n' + response
                
                # Step 4: Generate my next action
                hls_user_msg4 = self.generate_interaction_feedback_user_message4(time, possible_opponent_strategy)
                hls_user_msg = hls_user_msg + '\n\n' + hls_user_msg4
                responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg4])
                response = responses[0]
                my_next_strategy = self.extract_dict(response)
                
                return my_next_strategy
                
            else:
                # EXACT HM PATTERN: Use best hypothesis when good one found
                sorted_keys = sorted([key for key in self.opponent_hypotheses], 
                                   key=lambda x: self.opponent_hypotheses[x]['value'], reverse=True)
                best_key = sorted_keys[0]
                
                # EXACT HM PATTERN: Assert hypothesis meets threshold
                assert self.opponent_hypotheses[best_key]['value'] > self.good_hypothesis_thr
                
                print(f'Using good hypothesis: {best_key}')
                
                # Use best hypothesis for prediction and strategy
                best_hypothesis = self.opponent_hypotheses[best_key]
                
                hls_user_msg3 = self.generate_interaction_feedback_user_message3(time, best_hypothesis)
                responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg3])
                response = responses[0]
                other_player_next_action = self.extract_dict(response)
                
                # Update best hypothesis
                self.opponent_hypotheses[best_key]['other_player_next_action'] = other_player_next_action
                
                hls_user_msg4 = self.generate_interaction_feedback_user_message4(time, best_hypothesis)
                responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg4])
                response = responses[0]
                my_next_strategy = self.extract_dict(response)
                
                return my_next_strategy
        
        else:
            # EXACT HM PATTERN: Initial decision without prior interaction
            # Generate high-level strategy for initial action
            hls_user_msg = self.generate_hls_user_message(lob, time, countdown)
            responses = await self.controller.async_batch_prompt(self.system_message, [hls_user_msg])
            response = responses[0]
            my_strategy = self.extract_dict(response)
            
            return my_strategy

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
            
            Provide your response in Python dictionary format:
            ```python
            {{
              'my_next_quote_price': 125,
              'reasoning': 'Initial market analysis and strategy',
              'confidence': 0.7
            }}
            ```
            """
        return user_message

    def eval_hypotheses(self):
        """
        EXACT MIRROR: HM's eval_hypotheses method (lines 665-712)
        
        Evaluates opponent strategy hypotheses using exact Rescorla-Wagner update logic
        """
        if not self.opponent_hypotheses:
            return
            
        # EXACT HM PATTERN: Get latest key and sort others
        latest_key = max(self.opponent_hypotheses.keys())
        sorted_keys = sorted([key for key in self.opponent_hypotheses if key != latest_key],
                           key=lambda x: self.opponent_hypotheses[x]['value'], 
                           reverse=True)
        keys2eval = sorted_keys[:self.top_k] + [latest_key]
        
        # EXACT HM PATTERN: Reset good hypothesis flag
        self.good_hypothesis_found = False
        
        for key in keys2eval:
            # EXACT HM PATTERN: Evaluate prediction accuracy
            if 'other_player_next_action' not in self.opponent_hypotheses[key]:
                continue
                
            # Get predicted vs actual trading behavior
            predicted_data = self.opponent_hypotheses[key]['other_player_next_action']
            
            if 'predicted_other_trader_next_price' not in predicted_data:
                # Default to neutral prediction
                predicted_data['predicted_other_trader_next_price'] = 150
                
            predicted_price = predicted_data['predicted_other_trader_next_price']
            
            # Get actual competitor behavior from recent interactions
            if self.interaction_history and len(self.interaction_history) > 0:
                last_interaction = self.interaction_history[-1]
                actual_price = last_interaction.get('actual_competitor_price')
                
                if actual_price is not None:
                    # EXACT HM PATTERN: Binary evaluation of prediction accuracy
                    price_diff = abs(predicted_price - actual_price)
                    prediction_correct = price_diff < 10  # Trading threshold (similar to HM tie handling)
                    
                    # EXACT HM RESCORLA-WAGNER UPDATE
                    if prediction_correct:
                        prediction_error = self.correct_guess_reward - self.opponent_hypotheses[key]['value']
                    else:
                        prediction_error = -self.correct_guess_reward - self.opponent_hypotheses[key]['value']
                    
                    # Update value using exact HM formula
                    self.opponent_hypotheses[key]['value'] = self.opponent_hypotheses[key]['value'] + (self.alpha * prediction_error)
                    
                    # EXACT HM PATTERN: Check for good hypothesis
                    if self.opponent_hypotheses[key]['value'] > self.good_hypothesis_thr:
                        self.good_hypothesis_found = True

    def extract_dict(self, response: str) -> Dict[str, Any]:
        """
        EXACT MIRROR: HM's extract_dict method (lines 714-746)
        
        Extract dictionary from LLM response using exact same parsing logic
        """
        try:
            # EXACT HM PATTERN: Find JSON code markers
            start_marker = "```python\n"
            end_marker = "\n```"         
            start_pos = response.find(start_marker)
            
            if start_pos == -1:
                # Try alternative markers
                start_marker = "```python"
                start_pos = response.find(start_marker)
                if start_pos != -1:
                    start_index = start_pos + len(start_marker)
                    brace_start = response.find('{', start_index)
                    if brace_start != -1:
                        brace_count = 0
                        end_index = brace_start
                        for i, char in enumerate(response[brace_start:], brace_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_index = i + 1
                                    break
                        dict_str = response[brace_start:end_index].strip()
                    else:
                        raise ValueError("No dictionary found after ```python marker")
                else:
                    raise ValueError("Python dictionary markers not found in LLM response.")
            else:
                start_index = start_pos + len(start_marker)
                end_index = response.find(end_marker, start_index)
                if end_index == -1:
                    end_index = response.find("```", start_index)
                    if end_index == -1:
                        raise ValueError("Python dictionary end markers not found in LLM response.")
                dict_str = response[start_index: end_index].strip()

            # EXACT HM PATTERN: Process lines, skip comments
            lines = dict_str.split('\n')
            cleaned_lines = []
            for line in lines:
                comment_index = line.find('#')
                if comment_index != -1:
                    line = line[:comment_index].strip()
                if line:
                    cleaned_lines.append(line)

            cleaned_dict_str = ' '.join(cleaned_lines)
         
            # EXACT HM PATTERN: Convert to dictionary
            import ast
            extracted_dict = ast.literal_eval(cleaned_dict_str)
            return extracted_dict
            
        except Exception as e:
            print(f"Error parsing dictionary: {e}")
            print(f"Raw response: {response[:500]}")
            raise ValueError(f"LLM response parsing failed: {e}")

    # BSE Integration Methods
    
    def getorder(self, time: float, countdown: float, lob: Dict[str, Any]):
        """
        BSE integration point - uses HM two_level_plan for order generation
        """
        if len(self.orders) < 1:
            return None
            
        if not self.controller:
            return None
        
        try:
            # Determine if this is after a trade interaction
            after_interaction = (len(self.interaction_history) > 0 and 
                               time - self.interaction_history[-1].get('time', 0) < 1.0)
            
            # Use HM two_level_plan
            decision_result = asyncio.run(
                self.two_level_plan(lob, time, countdown, after_interaction)
            )
            
            # Extract quote price from HM decision
            quote_price = decision_result.get('my_next_quote_price', 150)
            
            # Create BSE order
            order = self.orders[0]
            quoteprice = int(max(1, min(500, quote_price)))
            
            # Ensure we don't violate limit price
            if order.otype == 'Bid' and quoteprice > order.price:
                quoteprice = order.price
            elif order.otype == 'Ask' and quoteprice < order.price:
                quoteprice = order.price
            
            new_order = Order(self.tid, order.otype, quoteprice, order.qty, time, lob['QID'])
            self.lastquote = new_order
            
            return new_order
            
        except Exception as e:
            print(f"HM order generation failed for {self.tid}: {e}")
            return None

    def respond(self, time: float, lob: Dict[str, Any], trade: Any, vrbs: bool):
        """
        BSE integration point - updates HM state on market events
        """
        # Update profit tracking
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        # Update HM memory states
        self.update_memory_states(lob, time)
        
        if trade:
            # Update HM interaction tracking
            self.interaction_history.append({
                'time': time,
                'event': 'trade',
                'trade': trade,
                'your_quote': self.lastquote.price if self.lastquote else None,
                'actual_competitor_price': trade.get('price') if isinstance(trade, dict) else None
            })
            self.interact_steps += 1
            self.interaction_num += 1
            
            # Log trade response
            if hasattr(self, 'trader_logger'):
                self.trader_logger.log_trade_response(trade, "Updated HM interaction history")
        
        return None