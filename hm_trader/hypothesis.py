"""
Hypothesis Management Module for Hypothetical-Minds Trading Agent

Handles opponent strategy hypothesis generation, evaluation, and belief updates
using the Rescorla-Wagner learning algorithm from Hypothetical-Minds.
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio


class HypothesisManager:
    """
    Manages opponent strategy hypotheses using Hypothetical-Minds approach.
    
    Tracks competitor behaviors, generates hypotheses about their strategies,
    and evaluates these hypotheses using Rescorla-Wagner learning.
    """
    
    def __init__(self, trader_id: str, alpha: float = 0.3, correct_guess_reward: float = 1.0, 
                 good_hypothesis_thr: float = 0.7, top_k: int = 5):
        """
        Initialize hypothesis manager
        
        Args:
            trader_id: Trader identifier
            alpha: Learning rate for Rescorla-Wagner updates
            correct_guess_reward: Reward for correct predictions
            good_hypothesis_thr: Threshold for considering hypothesis "good"
            top_k: Number of top hypotheses to evaluate
        """
        self.trader_id = trader_id
        self.alpha = alpha
        self.correct_guess_reward = correct_guess_reward
        self.good_hypothesis_thr = good_hypothesis_thr
        self.top_k = top_k
        
        # Hypothesis storage
        self.opponent_hypotheses: Dict[str, Dict[str, Any]] = {}
        self.interaction_num = 0
        self.good_hypothesis_found = False
        
        # History tracking
        self.interaction_history: List[Dict[str, Any]] = []
    
    def create_hypothesis_id(self) -> str:
        """Generate unique hypothesis ID"""
        return f"hyp_{self.trader_id}_{self.interaction_num:03d}"
    
    async def generate_hypothesis(self, llm_interface, system_message: str, 
                                recent_interactions: List[Dict[str, Any]], time: float) -> Dict[str, Any]:
        """
        Generate new opponent strategy hypothesis using LLM
        
        Args:
            llm_interface: LLM interface for generating hypothesis
            system_message: System prompt for LLM
            recent_interactions: Recent market interactions
            time: Current market time
            
        Returns:
            Generated hypothesis data
        """
        # Get top hypotheses for context
        top_hypotheses = self._get_top_hypotheses()
        
        # Generate hypothesis message
        hypothesis_msg = self._create_hypothesis_message(recent_interactions, top_hypotheses, time)
        
        try:
            # Get LLM response
            response = await llm_interface.get_response(system_message, hypothesis_msg)
            # Response is already extracted by the LLM controller
            hypothesis_data = response
            
            if 'opponent_strategy' not in hypothesis_data:
                raise ValueError(f"LLM response missing required 'opponent_strategy' key. Got: {hypothesis_data}")
            
            # Create hypothesis with HM structure
            hypothesis_id = self.create_hypothesis_id()
            hypothesis = {
                'id': hypothesis_id,
                'opponent_strategy': hypothesis_data['opponent_strategy'],
                'time_created': time,
                'value': 0.0,  # Starting value (will be updated by evaluation)
                'predictions': []  # Track predictions made
            }
            
            # Store hypothesis
            self.opponent_hypotheses[hypothesis_id] = hypothesis
            
            return hypothesis
            
        except Exception as e:
            raise RuntimeError(f"Hypothesis generation failed for {self.trader_id}: {e}")
    
    async def generate_prediction(self, llm_interface, system_message: str, 
                                hypothesis: Optional[Dict[str, Any]], time: float) -> Dict[str, Any]:
        """
        Generate prediction for opponent's next action
        
        Args:
            llm_interface: LLM interface
            system_message: System prompt
            hypothesis: Hypothesis to base prediction on (or None for general)
            time: Current market time
            
        Returns:
            Prediction data
        """
        prediction_msg = self._create_prediction_message(hypothesis, time)
        
        try:
            response = await llm_interface.get_response(system_message, prediction_msg)
            # Response is already extracted by the LLM controller
            prediction_data = response
            
            return prediction_data
            
        except Exception as e:
            raise RuntimeError(f"Prediction generation failed for {self.trader_id}: {e}")
    
    def evaluate_hypotheses(self) -> None:
        """
        Evaluate existing hypotheses using Rescorla-Wagner learning.
        EQUIVALENT: Exact HM evaluation logic from pd_hypothetical_minds.py
        """
        if not self.opponent_hypotheses or not self.interaction_history:
            return
        
        # Reset good hypothesis flag for this evaluation
        self.good_hypothesis_found = False
        
        # Get keys to evaluate (top K + latest)
        keys_list = list(self.opponent_hypotheses.keys())
        if not keys_list:
            return
        
        # Handle mixed key types by converting to strings for comparison
        if any(isinstance(k, int) for k in keys_list):
            int_keys = [k for k in keys_list if isinstance(k, int)]
            str_keys = [k for k in keys_list if isinstance(k, str)]
            latest_key = max(int_keys) if int_keys else str_keys[-1] if str_keys else keys_list[-1]
        else:
            latest_key = keys_list[-1]
        
        sorted_keys = sorted([key for key in self.opponent_hypotheses if key != latest_key],
                           key=lambda x: self.opponent_hypotheses[x].get('value', 0), 
                           reverse=True)
        keys2eval = sorted_keys[:self.top_k] + [latest_key]
        
        # Evaluate each hypothesis
        for key in keys2eval:
            if key not in self.opponent_hypotheses:
                continue
                
            hypothesis = self.opponent_hypotheses[key]
            
            # Get latest interaction for evaluation
            if self.interaction_history:
                last_interaction = self.interaction_history[-1]
                
                # Trading-specific hypothesis evaluation
                actual_price = last_interaction.get('actual_price')
                
                if actual_price and 'predicted_price' in hypothesis:
                    # Determine if prediction was correct
                    predicted_price = hypothesis['predicted_price']
                    price_diff = abs(predicted_price - actual_price)
                    
                    # Binary evaluation: correct if within threshold
                    prediction_correct = price_diff < 10  # Trading threshold
                    
                    # EXACT HM RESCORLA-WAGNER UPDATE
                    if prediction_correct:
                        prediction_error = self.correct_guess_reward - hypothesis['value']
                    else:
                        prediction_error = -self.correct_guess_reward - hypothesis['value']
                    
                    # Update value using HM formula
                    hypothesis['value'] = hypothesis['value'] + (self.alpha * prediction_error)
                    
                    # Check if hypothesis is now good
                    if hypothesis['value'] > self.good_hypothesis_thr:
                        self.good_hypothesis_found = True
    
    def get_best_hypothesis(self) -> Optional[Dict[str, Any]]:
        """
        Get the best hypothesis based on value
        
        Returns:
            Best hypothesis or None if no good hypothesis exists
        """
        if not self.good_hypothesis_found or not self.opponent_hypotheses:
            return None
        
        sorted_keys = sorted(self.opponent_hypotheses.keys(),
                           key=lambda x: self.opponent_hypotheses[x]['value'], 
                           reverse=True)
        
        best_key = sorted_keys[0]
        best_hypothesis = self.opponent_hypotheses[best_key]
        
        # Verify it meets the threshold
        if best_hypothesis['value'] > self.good_hypothesis_thr:
            return best_hypothesis
        
        return None
    
    def update_interaction_history(self, interaction: Dict[str, Any]) -> None:
        """
        Update interaction history
        
        Args:
            interaction: New interaction data
        """
        self.interaction_history.append(interaction)
        self.interaction_num += 1
        
        # Keep history manageable
        if len(self.interaction_history) > 50:
            self.interaction_history = self.interaction_history[-50:]
    
    def _get_top_hypotheses(self) -> Dict[str, Dict[str, Any]]:
        """Get top K hypotheses by value"""
        if not self.opponent_hypotheses:
            return {}
        
        sorted_keys = sorted(self.opponent_hypotheses.keys(),
                           key=lambda x: self.opponent_hypotheses[x]['value'], 
                           reverse=True)
        top_keys = sorted_keys[:self.top_k]
        
        return {key: self.opponent_hypotheses[key] for key in top_keys 
                if self.opponent_hypotheses[key]['value'] > 0}
    
    def _create_hypothesis_message(self, recent_interactions: List[Dict[str, Any]], 
                                 top_hypotheses: Dict[str, Dict[str, Any]], time: float) -> str:
        """Create LLM message for hypothesis generation"""
        return f"""
A trading interaction has occurred at time {time}. Recent interaction history: {recent_interactions}.
Previous hypotheses about opponent strategies: {top_hypotheses}.

STRATEGY: What is the opponent's likely trading strategy?
ANSWER: "They appear to be using [STRATEGY] strategy with [CONFIDENCE]% confidence."

Keep response short and direct.
"""
    
    def _create_prediction_message(self, hypothesis: Optional[Dict[str, Any]], time: float) -> str:
        """Create LLM message for prediction generation"""
        if hypothesis:
            strategy = hypothesis.get('opponent_strategy', 'general market participant')
            prompt_context = f"Based on your hypothesis that the opponent strategy is: {strategy}"
        else:
            prompt_context = "Based on general market analysis (no specific opponent hypothesis available)"
            
        recent_interaction = self.interaction_history[-1] if self.interaction_history else None
        
        return f"""
{prompt_context}
And the recent interaction: {recent_interaction}

PREDICT: What price will the opponent quote next?
ANSWER: "Competing traders will likely quote around [PRICE] with [CONFIDENCE]% confidence."

Keep response short and direct.
"""
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information for logging/debugging"""
        return {
            'num_hypotheses': len(self.opponent_hypotheses),
            'interaction_num': self.interaction_num,
            'good_hypothesis_found': self.good_hypothesis_found,
            'best_hypothesis_value': max([h['value'] for h in self.opponent_hypotheses.values()]) if self.opponent_hypotheses else 0.0
        }