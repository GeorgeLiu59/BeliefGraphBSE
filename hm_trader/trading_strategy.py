"""
Trading Strategy Module for Hypothetical-Minds Trading Agent

Handles trading decision logic, market data formatting, and strategy generation
using the two-level planning approach from Hypothetical-Minds.
"""

from typing import Dict, Any, Optional, Tuple
import asyncio


class TradingStrategy:
    """
    Manages trading strategy logic using Hypothetical-Minds two-level planning.
    
    Coordinates between hypothesis evaluation, prediction generation, and
    final trading decisions to maximize profit in the BSE market.
    """
    
    def __init__(self, trader_id: str, max_history: int = 50):
        """
        Initialize trading strategy manager
        
        Args:
            trader_id: Trader identifier
            max_history: Maximum trading history to keep
        """
        self.trader_id = trader_id
        self.max_history = max_history
        
        # Strategy state
        self.trading_history = []
        self.my_next_strategy = None
        self.current_strategy_context = None
    
    async def execute_two_level_plan(self, llm_interface, system_message: str, 
                                   hypothesis_manager, memory_manager,
                                   lob: Dict[str, Any], time: float, countdown: float, 
                                   after_trade: bool = False) -> Dict[str, Any]:
        """
        EQUIVALENT: Two-level planning from Hypothetical-Minds
        Main trading decision function with hypothesis generation and evaluation
        
        Args:
            llm_interface: LLM interface for strategy generation
            system_message: System prompt for LLM
            hypothesis_manager: Hypothesis management component
            memory_manager: Memory management component
            lob: Limit order book data
            time: Current market time
            countdown: Time remaining in session
            after_trade: Whether this is after a trade occurred
            
        Returns:
            Trading strategy decision
        """
        
        if after_trade:
            # EQUIVALENT TO HM: After interaction processing
            
            # Evaluate hypotheses if we have enough history
            if hypothesis_manager.interaction_num > 1:
                hypothesis_manager.evaluate_hypotheses()
            
            # Generate new hypothesis if no good one found
            if not hypothesis_manager.good_hypothesis_found:
                # Get recent interactions for context
                recent_interactions = memory_manager.get_interaction_context()[-3:]
                
                # Generate opponent strategy hypothesis
                hypothesis = await hypothesis_manager.generate_hypothesis(
                    llm_interface, system_message, recent_interactions, time
                )
                
                # Generate prediction for next opponent action
                prediction = await hypothesis_manager.generate_prediction(
                    llm_interface, system_message, hypothesis, time
                )
                
                # Store prediction in hypothesis
                hypothesis['opponent_next_action'] = prediction
                hypothesis['predicted_price'] = prediction.get('predicted_opponent_next_price', 150)
                
                # Generate my next action based on opponent prediction
                my_strategy = await self._generate_strategy_response(
                    llm_interface, system_message, hypothesis, time
                )
                
                self.my_next_strategy = my_strategy
                
            else:
                # Use best hypothesis when good one found
                best_hypothesis = hypothesis_manager.get_best_hypothesis()
                
                if best_hypothesis:
                    # Use best hypothesis for prediction
                    prediction = await hypothesis_manager.generate_prediction(
                        llm_interface, system_message, best_hypothesis, time
                    )
                    
                    # Update best hypothesis with new prediction
                    best_hypothesis['opponent_next_action'] = prediction
                    best_hypothesis['predicted_price'] = prediction.get('predicted_opponent_next_price', 150)
                    
                    # Generate strategy based on best hypothesis
                    my_strategy = await self._generate_strategy_response(
                        llm_interface, system_message, best_hypothesis, time
                    )
                    
                    self.my_next_strategy = my_strategy
        
        else:
            # Initial decision without prior interaction
            strategy = await self._generate_initial_strategy(
                llm_interface, system_message, lob, time, countdown
            )
            
            self.my_next_strategy = strategy
        
        return self.my_next_strategy
    
    async def _generate_initial_strategy(self, llm_interface, system_message: str,
                                       lob: Dict[str, Any], time: float, 
                                       countdown: float) -> Dict[str, Any]:
        """
        Generate initial trading strategy without hypothesis context
        
        Args:
            llm_interface: LLM interface
            system_message: System prompt
            lob: Limit order book data
            time: Current market time
            countdown: Time remaining
            
        Returns:
            Initial strategy decision
        """
        market_data = self.format_market_data(lob, time, countdown)
        trader_context = self.format_trader_context()
        
        strategy_msg = f"""
Current market state: {market_data}
Your trading context: {trader_context}

This is your initial trading decision. Analyze the market conditions and provide your trading strategy.

ABSOLUTELY CRITICAL - SYSTEM WILL CRASH IF NOT FOLLOWED:
- You MUST respond with ONLY the JSON format below
- NO explanations, NO additional text, NO markdown formatting
- ONLY the JSON block with ```json markers
- Any other text will cause immediate system failure

```json
{{
  "my_quote_price": 125,
  "reasoning": "Initial market analysis and strategy",
  "aggressiveness": "moderate"
}}
```
"""
        
        try:
            response = await llm_interface.get_response(system_message, strategy_msg)
            strategy_data = llm_interface.extract_json_dict(response)
            
            return strategy_data
            
        except Exception as e:
            raise RuntimeError(f"Initial strategy generation failed for {self.trader_id}: {e}")
    
    async def _generate_strategy_response(self, llm_interface, system_message: str,
                                        hypothesis: Dict[str, Any], time: float) -> Dict[str, Any]:
        """
        Generate strategy response based on hypothesis
        
        Args:
            llm_interface: LLM interface
            system_message: System prompt
            hypothesis: Opponent hypothesis to base strategy on
            time: Current market time
            
        Returns:
            Strategy decision
        """
        opponent_strategy = hypothesis.get('opponent_strategy', 'general market participant')
        predicted_next = hypothesis.get('opponent_next_action', {})
        
        strategy_msg = f"""
Given opponent strategy: {opponent_strategy}
Predicted opponent next action: {predicted_next}
Recent interaction history: {self.trading_history[-3:] if self.trading_history else []}

What should your next trading strategy be to maximize profit?
Consider:
1. Market trends and price movements
2. Long-term strategic positioning
3. Risk management

ABSOLUTELY CRITICAL - SYSTEM WILL CRASH IF NOT FOLLOWED:
- You MUST respond with ONLY the JSON format below
- NO explanations, NO additional text, NO markdown formatting
- ONLY the JSON block with ```json markers
- Any other text will cause immediate system failure

```json
{{
  "my_quote_price": 130,
  "reasoning": "Strategic reasoning for this price",
  "aggressiveness": "moderate"
}}
```
"""
        
        try:
            response = await llm_interface.get_response(system_message, strategy_msg)
            strategy_data = llm_interface.extract_json_dict(response)
            
            return strategy_data
            
        except Exception as e:
            raise RuntimeError(f"Strategy response generation failed for {self.trader_id}: {e}")
    
    def format_market_data(self, lob: Dict[str, Any], time: float, countdown: float) -> str:
        """
        Format market data for LLM consumption
        
        Args:
            lob: Limit order book data
            time: Current time
            countdown: Time remaining
            
        Returns:
            Formatted market data string
        """
        market_info = f"Time: {time}, Countdown: {countdown}\n"
        market_info += f"Best Bid: {lob.get('bids', {}).get('best', 'None')}\n"
        market_info += f"Best Ask: {lob.get('asks', {}).get('best', 'None')}\n"
        
        # Add recent tape entries
        if lob.get('tape') and len(lob['tape']) > 0:
            recent_trades = lob['tape'][-5:]  # Last 5 trades
            market_info += f"Recent trades: {recent_trades}\n"
        else:
            market_info += "Recent trades: []\n"
        
        return market_info
    
    def format_trader_context(self, trader_data: Dict[str, Any]) -> str:
        """
        Format trader's current context for LLM
        
        Args:
            trader_data: Dictionary containing trader state information
            
        Returns:
            Formatted trader context string
        """
        context = f"Trader ID: {trader_data.get('tid', 'Unknown')}\n"
        context += f"Balance: {trader_data.get('balance', 0)}\n"
        context += f"Trades: {trader_data.get('n_trades', 0)}\n"
        context += f"Profit per time: {trader_data.get('profitpertime', 0.0)}\n"
        context += f"Orders: {trader_data.get('orders', [])}\n"
        context += f"Last quote: {trader_data.get('lastquote', 'None')}\n"
        
        return context
    
    def log_decision(self, time: float, decision: Dict[str, Any], market_data: str = None) -> None:
        """
        Log trading decision to history
        
        Args:
            time: Market time of decision
            decision: Decision data
            market_data: Optional market data context
        """
        self.trading_history.append({
            'time': time,
            'decision': decision,
            'market_data': market_data
        })
        
        # Keep history manageable
        if len(self.trading_history) > self.max_history:
            self.trading_history = self.trading_history[-self.max_history:]
    
    def create_order_from_decision(self, decision: Dict[str, Any], order_template: Any, 
                                 time: float, qid: int) -> Optional[Any]:
        """
        Create order object from trading decision
        
        Args:
            decision: Trading decision containing quote_price
            order_template: Template order with type and limits
            time: Current time
            qid: Quote ID
            
        Returns:
            Order object or None if decision invalid
        """
        if decision.get('my_quote_price') is None:
            return None
        
        # Import Order class here to avoid circular imports
        from BSE import Order
        
        quoteprice = int(decision['my_quote_price'])
        
        # Ensure price is within valid range
        quoteprice = max(1, min(500, quoteprice))
        
        # Ensure we don't violate limit price
        if order_template.otype == 'Bid' and quoteprice > order_template.price:
            quoteprice = order_template.price
        elif order_template.otype == 'Ask' and quoteprice < order_template.price:
            quoteprice = order_template.price
        
        new_order = Order(self.trader_id, order_template.otype, quoteprice, 
                         order_template.qty, time, qid)
        
        return new_order
    
    def get_decision_context(self, lob: Dict[str, Any], time: float) -> Dict[str, Any]:
        """
        Get context for decision making
        
        Args:
            lob: Limit order book data
            time: Current market time
            
        Returns:
            Decision context dictionary
        """
        # Check if this is after a trade (simplified detection)
        after_trade = (len(self.trading_history) > 0 and 
                      time - self.trading_history[-1]['time'] < 1.0)
        
        return {
            'after_trade': after_trade,
            'recent_decisions': self.trading_history[-5:],
            'market_time': time,
            'has_prior_strategy': self.my_next_strategy is not None
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get strategy status for debugging/logging
        
        Returns:
            Status dictionary
        """
        return {
            'decisions_made': len(self.trading_history),
            'has_current_strategy': self.my_next_strategy is not None,
            'last_decision_time': self.trading_history[-1]['time'] if self.trading_history else None,
            'recent_quote_prices': [d['decision'].get('my_quote_price', 0) for d in self.trading_history[-5:]]
        }