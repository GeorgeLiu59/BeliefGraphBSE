#!/usr/bin/env python3
"""
Agent Attribute System for Self-Designing LLM Traders

This module allows LLM agents to design and adapt their own trading attributes
that influence how they interact with the belief graph and make trading decisions.
"""

import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum


class AttributeType(Enum):
    """Types of attributes that agents can design for themselves"""
    AGGRESSIVENESS = "aggressiveness"      # How aggressively to pursue trades
    RISK_TOLERANCE = "risk_tolerance"      # How much risk to accept
    PATIENCE = "patience"                  # How long to wait for optimal conditions
    ADAPTABILITY = "adaptability"          # How quickly to change strategies
    MOMENTUM_FOLLOWING = "momentum_following"  # How much to follow market trends
    MEAN_REVERSION = "mean_reversion"      # How much to bet on price reversals


@dataclass
class AttributeConstraint:
    """Constraints for attribute values"""
    min_value: float
    max_value: float
    description: str
    impact_on_trading: str


@dataclass
class AgentAttribute:
    """A single attribute with its value and metadata"""
    name: str
    value: float  # 0.0 to 1.0
    min_value: float
    max_value: float
    description: str
    impact_on_trading: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class AttributeSet:
    """Complete set of attributes for an agent"""
    aggressiveness: float
    risk_tolerance: float
    patience: float
    adaptability: float
    momentum_following: float
    mean_reversion: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate that all attributes are within valid ranges"""
        attributes = [
            self.aggressiveness, self.risk_tolerance, self.patience,
            self.adaptability, self.momentum_following, self.mean_reversion
        ]
        return all(0.0 <= attr <= 1.0 for attr in attributes)


class AttributeDesigner:
    """Handles the design of agent attributes"""
    
    # Attribute constraints and descriptions
    ATTRIBUTE_DEFINITIONS = {
        AttributeType.AGGRESSIVENESS: AttributeConstraint(
            min_value=0.0, max_value=1.0,
            description="How aggressively you pursue trading opportunities",
            impact_on_trading="Higher values = more aggressive bidding/asking, lower values = more conservative"
        ),
        AttributeType.RISK_TOLERANCE: AttributeConstraint(
            min_value=0.0, max_value=1.0,
            description="How much risk you're willing to accept in trades",
            impact_on_trading="Higher values = accept larger price swings, lower values = prefer safer trades"
        ),
        AttributeType.PATIENCE: AttributeConstraint(
            min_value=0.0, max_value=1.0,
            description="How long you wait for optimal trading conditions",
            impact_on_trading="Higher values = wait longer for better prices, lower values = trade more frequently"
        ),
        AttributeType.ADAPTABILITY: AttributeConstraint(
            min_value=0.0, max_value=1.0,
            description="How quickly you change strategies based on market conditions",
            impact_on_trading="Higher values = adapt quickly to changes, lower values = stick to initial strategy"
        ),
        AttributeType.MOMENTUM_FOLLOWING: AttributeConstraint(
            min_value=0.0, max_value=1.0,
            description="How much you follow market trends and momentum",
            impact_on_trading="Higher values = follow trends, lower values = ignore momentum"
        ),
        AttributeType.MEAN_REVERSION: AttributeConstraint(
            min_value=0.0, max_value=1.0,
            description="How much you bet on prices returning to average levels",
            impact_on_trading="Higher values = bet on reversals, lower values = follow trends"
        )
    }
    
    @classmethod
    def design_random(cls) -> AttributeSet:
        """Design random attributes for an agent"""
        return AttributeSet(
            aggressiveness=random.uniform(0.0, 1.0),
            risk_tolerance=random.uniform(0.0, 1.0),
            patience=random.uniform(0.0, 1.0),
            adaptability=random.uniform(0.0, 1.0),
            momentum_following=random.uniform(0.0, 1.0),
            mean_reversion=random.uniform(0.0, 1.0)
        )
    
    @classmethod
    def design_conservative(cls) -> AttributeSet:
        """Design conservative attributes (low risk, high patience)"""
        return AttributeSet(
            aggressiveness=random.uniform(0.0, 0.3),
            risk_tolerance=random.uniform(0.0, 0.3),
            patience=random.uniform(0.7, 1.0),
            adaptability=random.uniform(0.3, 0.6),
            momentum_following=random.uniform(0.2, 0.5),
            mean_reversion=random.uniform(0.6, 1.0)
        )
    
    @classmethod
    def design_aggressive(cls) -> AttributeSet:
        """Design aggressive attributes (high risk, low patience)"""
        return AttributeSet(
            aggressiveness=random.uniform(0.7, 1.0),
            risk_tolerance=random.uniform(0.7, 1.0),
            patience=random.uniform(0.0, 0.3),
            adaptability=random.uniform(0.6, 1.0),
            momentum_following=random.uniform(0.6, 1.0),
            mean_reversion=random.uniform(0.0, 0.3)
        )
    
    @classmethod
    def design_balanced(cls) -> AttributeSet:
        """Design balanced attributes (moderate values across all)"""
        return AttributeSet(
            aggressiveness=random.uniform(0.4, 0.6),
            risk_tolerance=random.uniform(0.4, 0.6),
            patience=random.uniform(0.4, 0.6),
            adaptability=random.uniform(0.4, 0.6),
            momentum_following=random.uniform(0.4, 0.6),
            mean_reversion=random.uniform(0.4, 0.6)
        )
    
    @classmethod
    def design_momentum(cls) -> AttributeSet:
        """Design momentum-following attributes"""
        return AttributeSet(
            aggressiveness=random.uniform(0.5, 0.8),
            risk_tolerance=random.uniform(0.5, 0.8),
            patience=random.uniform(0.2, 0.5),
            adaptability=random.uniform(0.6, 1.0),
            momentum_following=random.uniform(0.8, 1.0),
            mean_reversion=random.uniform(0.0, 0.2)
        )
    
    @classmethod
    def design_mean_reversion(cls) -> AttributeSet:
        """Design mean-reversion attributes"""
        return AttributeSet(
            aggressiveness=random.uniform(0.3, 0.6),
            risk_tolerance=random.uniform(0.4, 0.7),
            patience=random.uniform(0.6, 1.0),
            adaptability=random.uniform(0.3, 0.6),
            momentum_following=random.uniform(0.0, 0.3),
            mean_reversion=random.uniform(0.8, 1.0)
        )
    
    @classmethod
    def get_design_strategies(cls) -> List[str]:
        """Get list of available design strategies"""
        return ["random", "conservative", "aggressive", "balanced", "momentum", "mean_reversion"]
    
    @classmethod
    def design_attributes(cls, strategy: str) -> AttributeSet:
        """Design attributes using the specified strategy"""
        strategy_map = {
            "random": cls.design_random,
            "conservative": cls.design_conservative,
            "aggressive": cls.design_aggressive,
            "balanced": cls.design_balanced,
            "momentum": cls.design_momentum,
            "mean_reversion": cls.design_mean_reversion
        }
        
        if strategy not in strategy_map:
            raise ValueError(f"Unknown design strategy: {strategy}")
        
        return strategy_map[strategy]()


class AttributeAdapter:
    """Handles adaptation of agent attributes based on performance and market conditions"""
    
    def __init__(self, current_attributes: AttributeSet):
        self.current_attributes = current_attributes
        self.adaptation_history: List[Dict[str, Any]] = []
    
    def should_adapt(self, performance_metrics: Dict[str, Any]) -> bool:
        """Determine if attributes should be adapted"""
        # Adapt if performance is poor
        if performance_metrics.get('profit', 0) < -50:  # Lost more than $50
            return True
        
        # Adapt if market conditions have changed significantly
        if performance_metrics.get('market_volatility', 0) > 0.8:
            return True
        
        # Adapt if other agents are outperforming significantly
        if performance_metrics.get('relative_performance', 0) < -0.2:
            return True
        
        return False
    
    def determine_adaptation_strategy(self, performance: Dict[str, Any]) -> str:
        """Determine which adaptation strategy to use"""
        if performance.get('profit', 0) < -100:
            return "conservative"  # Big losses -> become more conservative
        elif performance.get('market_volatility', 0) > 0.8:
            return "balanced"      # High volatility -> become more balanced
        elif performance.get('relative_performance', 0) < -0.3:
            return "aggressive"    # Underperforming -> become more aggressive
        else:
            return "balanced"      # Default to balanced
    
    def apply_adaptation(self, strategy: str, reasoning: str, performance_metrics: Dict[str, Any] = None) -> AttributeSet:
        """Apply adaptation strategy to current attributes"""
        # Get new base attributes
        new_attributes = AttributeDesigner.design_attributes(strategy)
        
        # Blend with current attributes based on adaptability
        blend_factor = self.current_attributes.adaptability
        
        adapted_attributes = AttributeSet(
            aggressiveness=self._blend_attribute(
                self.current_attributes.aggressiveness,
                new_attributes.aggressiveness,
                blend_factor
            ),
            risk_tolerance=self._blend_attribute(
                self.current_attributes.risk_tolerance,
                new_attributes.risk_tolerance,
                blend_factor
            ),
            patience=self._blend_attribute(
                self.current_attributes.patience,
                new_attributes.patience,
                blend_factor
            ),
            adaptability=self._blend_attribute(
                self.current_attributes.adaptability,
                new_attributes.adaptability,
                blend_factor
            ),
            momentum_following=self._blend_attribute(
                self.current_attributes.momentum_following,
                new_attributes.momentum_following,
                blend_factor
            ),
            mean_reversion=self._blend_attribute(
                self.current_attributes.mean_reversion,
                new_attributes.mean_reversion,
                blend_factor
            )
        )
        
        # Record adaptation
        self.adaptation_history.append({
            'timestamp': performance_metrics.get('timestamp', 0) if performance_metrics else time.time(),
            'old_attributes': self.current_attributes.to_dict(),
            'new_attributes': adapted_attributes.to_dict(),
            'strategy': strategy,
            'reasoning': reasoning,
            'performance': performance_metrics or {}
        })
        
        # Update current attributes
        self.current_attributes = adapted_attributes
        
        return adapted_attributes
    
    def _blend_attribute(self, current: float, target: float, blend_factor: float) -> float:
        """Blend current and target attribute values"""
        return current * (1 - blend_factor) + target * blend_factor
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of attribute adaptations"""
        return self.adaptation_history


class AttributeManager:
    """Main class for managing agent attributes"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.attributes: Optional[AttributeSet] = None
        self.adapter: Optional[AttributeAdapter] = None
        self.design_strategy: str = "balanced"
        self.adaptation_enabled: bool = True
    
    def initialize_attributes(self, strategy: str = "balanced") -> AttributeSet:
        """Initialize agent attributes using the specified strategy"""
        self.design_strategy = strategy
        self.attributes = AttributeDesigner.design_attributes(strategy)
        self.adapter = AttributeAdapter(self.attributes)
        return self.attributes
    
    def get_attributes(self) -> AttributeSet:
        """Get current attributes"""
        if self.attributes is None:
            raise ValueError("Attributes not initialized. Call initialize_attributes() first.")
        return self.attributes
    
    def adapt_attributes(self, performance_metrics: Dict[str, Any]) -> Optional[AttributeSet]:
        """Adapt attributes based on performance if adaptation is enabled"""
        if not self.adaptation_enabled or self.adapter is None:
            return None
        
        if self.adapter.should_adapt(performance_metrics):
            strategy = self.adapter.determine_adaptation_strategy(performance_metrics)
            reasoning = f"Performance: {performance_metrics.get('profit', 0)}, Market volatility: {performance_metrics.get('market_volatility', 0)}"
            return self.adapter.apply_adaptation(strategy, reasoning)
        
        return None
    
    def to_json(self) -> str:
        """Convert attribute manager to JSON"""
        data = {
            'agent_id': self.agent_id,
            'attributes': self.attributes.to_dict() if self.attributes else None,
            'design_strategy': self.design_strategy,
            'adaptation_enabled': self.adaptation_enabled,
            'adaptation_history': self.adapter.get_adaptation_history() if self.adapter else []
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AttributeManager':
        """Create attribute manager from JSON"""
        data = json.loads(json_str)
        manager = cls(data['agent_id'])
        manager.design_strategy = data['design_strategy']
        manager.adaptation_enabled = data['adaptation_enabled']
        
        if data['attributes']:
            manager.attributes = AttributeSet(**data['attributes'])
            manager.adapter = AttributeAdapter(manager.attributes)
            
            # Restore adaptation history
            for history_item in data['adaptation_history']:
                if manager.adapter:
                    manager.adapter.adaptation_history.append(history_item)
        
        return manager
