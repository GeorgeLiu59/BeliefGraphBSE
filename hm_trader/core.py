"""
Core Trader Module for Hypothetical-Minds Trading Agent

Clean, modular implementation of TraderLLM_HM that orchestrates:
- LLM interactions via llm_controller
- Hypothesis management via hypothesis module
- Memory management via memory module  
- Trading strategy via trading_strategy module

Maintains exact business logic while improving code structure and readability.
"""

import os
import asyncio
from typing import Dict, Any, Optional
from queue import Queue

# Import components
from .llm_controller import LLMInterface, create_system_message
from .hypothesis import HypothesisManager
from .memory import MemoryManager
from .trading_strategy import TradingStrategy


class TraderLLM_HM:
    """
    Hypothetical-Minds LLM trader using clean modular architecture.
    
    This is the main trader class that coordinates between all components
    to implement the HM trading strategy while maintaining the exact same
    business logic as the original implementation.
    
    Flow:
    (1) initialize → (2) env setup → (3) agent create → (4) game loop {
        (4a) env.reset() → (4b) agent.act() → (4c) two_level_plan → 
        (4d) get_actions_from_plan → (4e) env.step() → 
        (4f) IF trade: execution → (4g) learning → (4h) hypothesis building → 
        (4i) next iteration
    }
    """
    
    def __init__(self, ttype: str, tid: str, balance: float, params: Optional[Dict[str, Any]], time: float):
        """
        (3) AGENT CREATE - Initialize the Hypothetical-Minds LLM trader
        
        Args:
            ttype: Trader type
            tid: Trader ID
            balance: Starting balance
            params: Configuration parameters
            time: Current time
        """
        # Import Trader base class here to avoid circular imports
        from BSE import Trader
        Trader.__init__(self, ttype, tid, balance, params, time)
         
        # Configuration
        self.config = {'agent_id': tid, 'self_improve': params.get('self_improve', True) if params else True}
        self.agent_id = tid
        self.debug_mode = params.get('debug_mode', False) if params else False
        
        # Initialize logger first
        from logging_config import TraderLogger
        self.trader_logger = TraderLogger(tid, ttype)
        
        # HM parameters (must be set before initializing components)
        self.alpha = 0.3
        self.correct_guess_reward = 1.0
        self.good_hypothesis_thr = 0.7
        self.top_k = 5
        self.self_improve = self.config['self_improve']
        self.max_history = 50
        
        # Initialize core components
        self._initialize_components()
        
        # Trading state
        self.all_actions = Queue()
        self.system_message = create_system_message(tid)
        
        # History tracking
        self.interact_steps = 0
        
        self.trader_logger.print_style_log('info', 
            f"Initialized clean modular LLM_HM trader {tid}")
    
    def _initialize_components(self) -> None:
        """Initialize all modular components"""
        # LLM interface
        self.llm_interface = LLMInterface(self.tid)
        
        # Hypothesis manager
        self.hypothesis_manager = HypothesisManager(
            self.tid, self.alpha, self.correct_guess_reward, 
            self.good_hypothesis_thr, self.top_k
        )
        
        # Memory manager
        self.memory_manager = MemoryManager(self.tid, self.max_history)
        
        # Trading strategy
        self.trading_strategy = TradingStrategy(self.tid, self.max_history)
    
    def getorder(self, time: float, countdown: float, lob: Dict[str, Any]) -> Optional[Any]:
        """
        (4b) AGENT.ACT() + (4c) TWO_LEVEL_PLAN + (4d) GET_ACTIONS_FROM_PLAN
        Create this trader's order to be sent to the exchange
        
        Args:
            time: Current time
            countdown: Time remaining
            lob: Limit order book data
            
        Returns:
            New order or None
        """
        if len(self.orders) < 1:
            return None
        
        if not self.llm_interface.is_available():
            return None
        
        try:
            # (4b) AGENT.ACT() - Use existing knowledge to make trading decisions
            # (4c) TWO_LEVEL_PLAN - Execute HM-style trading decision plan
            
            # Get decision context
            decision_context = self.trading_strategy.get_decision_context(lob, time)
            
            # Execute two-level planning
            decision_result = asyncio.run(
                self.trading_strategy.execute_two_level_plan(
                    self.llm_interface, self.system_message,
                    self.hypothesis_manager, self.memory_manager,
                    lob, time, countdown, decision_context['after_trade']
                )
            )
            
            # Extract decision
            decision = {
                'quote_price': decision_result.get('my_quote_price', 150),
                'reasoning': decision_result.get('reasoning', 'HM strategy decision'),
                'confidence': 0.8
            }
            
            # Log decision
            market_data = self.trading_strategy.format_market_data(lob, time, countdown)
            self.trading_strategy.log_decision(time, decision, market_data)
            
            # (4d) GET_ACTIONS_FROM_PLAN - Create order based on decision
            order = self.orders[0]
            new_order = self.trading_strategy.create_order_from_decision(
                decision, order, time, lob['QID']
            )
            
            if new_order:
                self.lastquote = new_order
                
                if self.debug_mode:
                    self.trader_logger.print_style_log('debug', 
                        f"ORDER EXECUTION | Trader: {self.tid} | Decision Price: {decision['quote_price']} | Final Price: {new_order.price} | Type: {order.otype}")
            
            return new_order
            
        except Exception as e:
            self.trader_logger.print_style_log('error', 
                f"Order generation failed | Trader: {self.tid} | Error: {e}")
            return None
    
    def respond(self, time: float, lob: Dict[str, Any], trade: Any, vrbs: bool) -> None:
        """
        (4f) IF TRADE: EXECUTION + (4g) LEARNING + (4h) HYPOTHESIS BUILDING
        Respond to market events and update agent state
        
        Args:
            time: Current time
            lob: Limit order book
            trade: Recent trade (if any)
            vrbs: Verbosity flag
        """
        # Update profit per time
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        
        # (4g) LEARNING - Update memory with current market state
        self.memory_manager.update_memory(lob, time)
        
        # Log trade and update interaction history
        if trade:
            self.interact_steps += 1
            
            # Update interaction history for hypothesis evaluation
            self.hypothesis_manager.update_interaction_history({
                'time': time,
                'event': 'trade',
                'trade': trade,
                'actual_price': trade['price'] if isinstance(trade, dict) and 'price' in trade else None
            })
            
            # (4h) HYPOTHESIS BUILDING - Generate and evaluate hypotheses
            if self.self_improve and self.hypothesis_manager.interaction_num > 1:
                try:
                    # Generate opponent hypotheses
                    asyncio.run(self._generate_opponent_hypotheses(lob, time))
                    
                    # Evaluate existing hypotheses
                    self.hypothesis_manager.evaluate_hypotheses()
                    
                except Exception as e:
                    self.trader_logger.print_style_log('error', 
                        f"Hypothesis processing failed | Trader: {self.tid} | Error: {e}")
            
            self.trader_logger.log_trade_response(trade, f"Updated hypotheses and memory")
    
    async def _generate_opponent_hypotheses(self, lob: Dict[str, Any], time: float) -> None:
        """
        Generate hypotheses about opponent strategies using LLM reasoning
        
        Args:
            lob: Limit order book data
            time: Current market time
        """
        # Get interaction context
        interaction_context = self.memory_manager.get_interaction_context()
        
        if len(interaction_context) < 3:
            return  # Need some history to generate hypotheses
        
        try:
            # Generate hypothesis using the hypothesis manager
            hypothesis = await self.hypothesis_manager.generate_hypothesis(
                self.llm_interface, self.system_message, 
                interaction_context[-10:], time
            )
            
            # Generate prediction for the hypothesis
            prediction = await self.hypothesis_manager.generate_prediction(
                self.llm_interface, self.system_message, hypothesis, time
            )
            
            # Update hypothesis with prediction
            hypothesis['predicted_price'] = prediction.get('predicted_opponent_next_price', 150)
            hypothesis['opponent_next_action'] = prediction
            
            if self.debug_mode:
                self.trader_logger.print_style_log('debug', 
                    f"HYPOTHESIS GENERATED | Trader: {self.tid} | Strategy: {hypothesis['opponent_strategy'][:50]}... | Predicted Price: {hypothesis['predicted_price']}")
            
        except Exception as e:
            self.trader_logger.print_style_log('error', 
                f"Hypothesis generation failed | Trader: {self.tid} | Error: {e}")
    
    def extract_dict(self, response: str) -> Dict[str, Any]:
        """
        Extract dictionary from LLM response using HM parsing logic
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed dictionary
            
        Raises:
            ValueError: If parsing fails
        """
        return self.llm_interface.extract_json_dict(response)
    
    def _get_trader_data(self) -> Dict[str, Any]:
        """
        Get trader data for strategy formatting
        
        Returns:
            Dictionary containing trader state
        """
        return {
            'tid': self.tid,
            'balance': self.balance,
            'n_trades': self.n_trades,
            'profitpertime': self.profitpertime,
            'orders': self.orders,
            'lastquote': self.lastquote
        }
    
    def _format_trader_context(self) -> str:
        """
        Format trader's current context for the LLM
        
        Returns:
            Formatted trader context string
        """
        trader_data = self._get_trader_data()
        return self.trading_strategy.format_trader_context(trader_data)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status for debugging/monitoring
        
        Returns:
            Status dictionary with all component states
        """
        return {
            'trader_id': self.tid,
            'llm_available': self.llm_interface.is_available(),
            'hypothesis_status': self.hypothesis_manager.get_status(),
            'memory_status': self.memory_manager.get_status(),
            'strategy_status': self.trading_strategy.get_status(),
            'balance': self.balance,
            'n_trades': self.n_trades,
            'interact_steps': self.interact_steps
        }
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self, 'memory_manager'):
            self.memory_manager.clear_memory()
        
        # Clear trading history
        if hasattr(self, 'trading_strategy'):
            self.trading_strategy.trading_history.clear()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during deletion