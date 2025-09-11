"""
Memory Management Module for Hypothetical-Minds Trading Agent

Handles market state tracking, competitor action monitoring, and memory state updates
following the Hypothetical-Minds memory pattern for entity tracking.
"""

from typing import Dict, List, Any, Optional, Tuple


class MemoryManager:
    """
    Manages memory states for HM trading agent.
    
    Tracks market conditions, competitor actions, and price levels
    to support hypothesis generation and decision making.
    """
    
    def __init__(self, trader_id: str, max_history: int = 100):
        """
        Initialize memory manager
        
        Args:
            trader_id: Trader identifier
            max_history: Maximum number of historical entries to keep
        """
        self.trader_id = trader_id
        self.max_history = max_history
        
        # Memory states following HM pattern
        self.memory_states = {
            'market_state': [],      # Market conditions over time
            'competitor_actions': [], # Competitor trading actions
            'price_levels': []       # Price level history
        }
    
    def update_market_state(self, lob: Dict[str, Any], time: float) -> None:
        """
        Update market state memory with latest LOB data
        
        Args:
            lob: Limit order book data
            time: Current market time
        """
        market_state = {
            'time': time,
            'best_bid': lob.get('bids', {}).get('best'),
            'best_ask': lob.get('asks', {}).get('best'),
            'spread': None
        }
        
        # Calculate spread if both bid and ask are available
        if market_state['best_bid'] and market_state['best_ask']:
            market_state['spread'] = market_state['best_ask'] - market_state['best_bid']
        
        # Add to memory using HM tuple format: (observation, step_info, distance_info)
        self.memory_states['market_state'].append((
            market_state, 
            f'Step: {int(time)}', 
            'distance: current'
        ))
        
        self._maintain_memory_size('market_state')
    
    def update_competitor_actions(self, lob: Dict[str, Any], time: float) -> None:
        """
        Update competitor action tracking from market tape
        
        Args:
            lob: Limit order book data containing tape
            time: Current market time
        """
        if not lob.get('tape') or len(lob['tape']) == 0:
            return
        
        # Extract recent trades from tape
        recent_trades = lob['tape'][-5:]  # Last 5 trades
        competitor_actions = []
        
        for trade in recent_trades:
            if (trade.get('type') == 'Trade' and 
                trade.get('price') and 
                trade['price'] != 'None'):
                
                competitor_actions.append({
                    'time': trade.get('time', time),
                    'price': trade['price'],
                    'party1': trade.get('party1'),
                    'party2': trade.get('party2'),
                    'action_type': 'trade'
                })
        
        if competitor_actions:
            # Add each action to memory
            for action in competitor_actions:
                self.memory_states['competitor_actions'].append((
                    action,
                    f'Trade at {action["time"]:.2f}',
                    'distance: recent'
                ))
            
            self._maintain_memory_size('competitor_actions')
    
    def update_price_levels(self, lob: Dict[str, Any], time: float) -> None:
        """
        Track price level movements
        
        Args:
            lob: Limit order book data
            time: Current market time
        """
        best_bid = lob.get('bids', {}).get('best')
        best_ask = lob.get('asks', {}).get('best')
        
        # Add price levels to tracking
        if best_bid:
            self.memory_states['price_levels'].append(best_bid)
        if best_ask:
            self.memory_states['price_levels'].append(best_ask)
        
        self._maintain_memory_size('price_levels')
    
    def update_memory(self, lob: Dict[str, Any], time: float) -> None:
        """
        EQUIVALENT: Comprehensive memory update following HM pattern
        Updates all memory components with latest market data
        
        Args:
            lob: Limit order book data
            time: Current market time
        """
        # Remove older entries for the same time (HM pattern)
        self._clean_duplicate_times(time)
        
        # Update all memory components
        self.update_market_state(lob, time)
        self.update_competitor_actions(lob, time)
        self.update_price_levels(lob, time)
    
    def get_recent_market_context(self, n_recent: int = 10) -> Dict[str, Any]:
        """
        Get recent market context for decision making
        
        Args:
            n_recent: Number of recent entries to include
            
        Returns:
            Dictionary containing recent market context
        """
        context = {
            'recent_market_states': self.memory_states['market_state'][-n_recent:],
            'recent_competitor_actions': self.memory_states['competitor_actions'][-n_recent:],
            'recent_price_levels': self.memory_states['price_levels'][-n_recent:],
            'market_trend': self._analyze_price_trend(),
            'competitor_activity_level': len(self.memory_states['competitor_actions'][-n_recent:])
        }
        
        return context
    
    def get_interaction_context(self) -> List[Dict[str, Any]]:
        """
        Get context for hypothesis generation based on recent interactions
        
        Returns:
            List of recent interaction contexts
        """
        interaction_context = []
        
        # Get recent competitor actions and market states
        recent_actions = self.memory_states['competitor_actions'][-10:]
        recent_market_states = self.memory_states['market_state'][-10:]
        
        # Combine actions with corresponding market states
        for i, action_tuple in enumerate(recent_actions):
            action = action_tuple[0] if isinstance(action_tuple, tuple) else action_tuple
            
            # Find corresponding market state
            market_state = None
            if i < len(recent_market_states):
                market_state_tuple = recent_market_states[i]
                market_state = market_state_tuple[0] if isinstance(market_state_tuple, tuple) else market_state_tuple
            
            interaction_context.append({
                'time': action.get('time'),
                'competitor_price': action.get('price'),
                'market_state': market_state
            })
        
        return interaction_context
    
    def _maintain_memory_size(self, memory_type: str) -> None:
        """
        Keep memory size manageable for specific memory type
        
        Args:
            memory_type: Type of memory to maintain
        """
        if memory_type in self.memory_states:
            if len(self.memory_states[memory_type]) > self.max_history:
                self.memory_states[memory_type] = self.memory_states[memory_type][-self.max_history:]
    
    def _clean_duplicate_times(self, current_time: float) -> None:
        """
        Remove older references of same time (HM pattern)
        
        Args:
            current_time: Current time to check for duplicates
        """
        for memory_type in self.memory_states:
            self.memory_states[memory_type] = [
                obs for obs in self.memory_states[memory_type] 
                if not (
                    (isinstance(obs, tuple) and len(obs) >= 1 and 
                     isinstance(obs[0], dict) and obs[0].get('time') == current_time) or
                    (isinstance(obs, dict) and obs.get('time') == current_time)
                )
            ]
    
    def _analyze_price_trend(self) -> str:
        """
        Analyze recent price trend for context
        
        Returns:
            Trend description: 'rising', 'falling', 'stable', or 'unknown'
        """
        if len(self.memory_states['price_levels']) < 5:
            return 'unknown'
        
        recent_prices = self.memory_states['price_levels'][-5:]
        
        # Simple trend analysis
        if len(set(recent_prices)) == 1:
            return 'stable'
        
        # Check if generally rising or falling
        first_half = recent_prices[:len(recent_prices)//2]
        second_half = recent_prices[len(recent_prices)//2:]
        
        if sum(second_half) > sum(first_half):
            return 'rising'
        elif sum(second_half) < sum(first_half):
            return 'falling'
        else:
            return 'stable'
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get memory status for debugging/logging
        
        Returns:
            Status dictionary
        """
        return {
            'market_states_count': len(self.memory_states['market_state']),
            'competitor_actions_count': len(self.memory_states['competitor_actions']),
            'price_levels_count': len(self.memory_states['price_levels']),
            'price_trend': self._analyze_price_trend(),
            'total_memory_entries': sum(len(v) for v in self.memory_states.values())
        }
    
    def clear_memory(self) -> None:
        """Clear all memory states"""
        for memory_type in self.memory_states:
            self.memory_states[memory_type].clear()
    
    def get_memory_summary(self) -> str:
        """
        Get formatted memory summary for LLM context
        
        Returns:
            Formatted string summarizing memory state
        """
        status = self.get_status()
        recent_context = self.get_recent_market_context(5)
        
        summary = f"""
Market Memory Summary:
- Market states tracked: {status['market_states_count']}
- Competitor actions tracked: {status['competitor_actions_count']}
- Price trend: {status['price_trend']}
- Recent activity level: {status['competitor_actions_count']} actions

Recent Context:
{recent_context}
"""
        return summary.strip()