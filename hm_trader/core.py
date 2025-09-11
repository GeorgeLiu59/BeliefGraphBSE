"""
Core Trader Module for Hypothetical-Minds Trading Agent

EXACT MIRROR of Hypothetical-Minds structure adapted for BSE.
Uses the authentic HM DecentralizedAgent pattern with minimal adaptation.
"""

import os
from typing import Dict, Any, Optional

# Import HM controller
from .llm_controller import LLMInterface


class TraderLLM_HM:
    """
    BSE integration wrapper for authentic Hypothetical-Minds DecentralizedAgent.
    
    This class provides BSE compatibility while delegating core logic 
    to the authentic HM implementation in bse_hypothetical_minds.py
    """
    
    # HYPOTHETICAL-MINDS AGENT FLOW (UPDATED FOR BSE SEPARATION):
    # (1) initialize → (2) env setup → (3) agent create → (4) trading loop {
    #     (4a) BSE market reset → (4b) agent.act() → (4c) strategic_planning → (4d) order_creation → 
    #     (4e) market.step() → (4f) IF trade: respond() → (4g) learning → (4h) hypothesis_update → (4i) next_iteration
    # }
    #
    # STEP LOCATIONS (DESIGN CHANGE: SEPARATED LEARNING FROM ACTION):
    # (1) initialize: hm_trader/core.py → TraderLLM_HM.__init__()
    # (2) env setup: hm_trader/llm_controller.py → LLMInterface._initialize_controller()
    # (3) agent create: hm_trader/bse_trading/bse_hypothetical_minds.py → DecentralizedAgent.__init__()
    # (4a) market reset: BSE.py → market session initialization
    # (4b) agent.act(): hm_trader/bse_trading/bse_hypothetical_minds.py → DecentralizedAgent.getorder()
    # (4c) strategic_planning: hm_trader/bse_trading/bse_hypothetical_minds.py → plan_action_with_hypotheses()
    # (4d) order_creation: hm_trader/bse_trading/bse_hypothetical_minds.py → BSE order generation logic
    # (4e) market.step(): BSE.py → order matching and trade execution in market
    # (4f) respond(): hm_trader/bse_trading/bse_hypothetical_minds.py → DecentralizedAgent.respond()
    # (4g) learning: hm_trader/bse_trading/bse_hypothetical_minds.py → learn_from_interaction()
    # (4h) hypothesis_update: hm_trader/bse_trading/bse_hypothetical_minds.py → eval_hypotheses() + opponent modeling
    # (4i) next_iteration: BSE.py → continue trading loop
    #
    # KEY DESIGN CHANGE: Original HM combined learning+action in two_level_plan().
    # BSE requires separation: getorder() = pure action, respond() = pure learning.
    
    def __init__(self, ttype: str, tid: str, balance: float, params: Optional[Dict[str, Any]], time: float):
        """
        (1) INITIALIZE: Set up HM trader with BSE compatibility
        
        INTUITION: This is the birth of an HM agent. We create a wrapper that connects
        BSE's trading interface to the authentic Hypothetical-Minds reasoning system.
        The agent gets its identity, starting resources, and core configuration.
        
        Args:
            ttype: Trader type 
            tid: Trader ID
            balance: Starting balance
            params: Configuration parameters
            time: Current time
        """
        # Create HM-style config
        config = {
            'agent_id': tid,
            'self_improve': params.get('self_improve', True) if params else True
        }
        
        # Initialize LLM controller 
        llm_interface = LLMInterface(tid)
        
        # Log the initialization step
        import logging
        hm_logger = logging.getLogger('BSE.LLM_HM')
        hm_logger.info(f"[{tid}] STAGE (1) INITIALIZE | Creating TraderLLM_HM wrapper for HM agent")
        
        # Create authentic HM DecentralizedAgent
        from .bse_trading.bse_hypothetical_minds import DecentralizedAgent
        self.hm_agent = DecentralizedAgent(
            ttype=ttype,
            tid=tid, 
            balance=balance,
            params=params,
            time=time,
            config=config,
            controller=llm_interface if llm_interface.is_available() else None
        )
        
        # Expose BSE-required attributes by delegation
        self.tid = self.hm_agent.tid
        self.ttype = self.hm_agent.ttype
        
    def __getattr__(self, name):
        """Delegate all other attributes to HM agent"""
        return getattr(self.hm_agent, name)
    
    def getorder(self, time: float, countdown: float, lob: Dict[str, Any]):
        """Delegate to HM agent's getorder method"""
        return self.hm_agent.getorder(time, countdown, lob)
    
    def respond(self, time: float, lob: Dict[str, Any], trade: Any, vrbs: bool):
        """Delegate to HM agent's respond method"""
        return self.hm_agent.respond(time, lob, trade, vrbs)
    
    def add_order(self, order, vrbs: bool = False):
        """Delegate to HM agent's add_order method"""
        return self.hm_agent.add_order(order, vrbs)
    
    def del_order(self, order):
        """Delegate to HM agent's del_order method"""  
        return self.hm_agent.del_order(order)
    
    def bookkeep(self, time: float, trade: Any, order: Any, vrbs: bool):
        """Delegate to HM agent's bookkeep method"""
        return self.hm_agent.bookkeep(time, trade, order, vrbs)
    
    def profitpertime_update(self, time: float, birthtime: float, balance: float):
        """Delegate to HM agent's profitpertime_update method"""
        return self.hm_agent.profitpertime_update(time, birthtime, balance)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status including HM-specific data"""
        return {
            'trader_id': self.tid,
            'trader_type': self.ttype,
            'llm_available': hasattr(self.hm_agent, 'controller') and self.hm_agent.controller is not None,
            'balance': self.hm_agent.balance,
            'n_trades': self.hm_agent.n_trades,
            'interact_steps': getattr(self.hm_agent, 'interact_steps', 0),
            'interaction_num': getattr(self.hm_agent, 'interaction_num', 0),
            'good_hypothesis_found': getattr(self.hm_agent, 'good_hypothesis_found', False),
            'num_hypotheses': len(getattr(self.hm_agent, 'opponent_hypotheses', {}))
        }
        
    def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self.hm_agent, 'cleanup'):
            self.hm_agent.cleanup()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass