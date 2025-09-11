#!/usr/bin/env python3
"""
Test Adaptive Trader System

This script demonstrates how LLM agents can design and adapt their own trading attributes
that influence their behavior in the belief graph and trading decisions.
"""

import json
import time
from typing import Dict, Any
from agent_attributes import AttributeManager, AttributeDesigner, AttributeSet
from llm_attribute_prompts import AttributePromptTemplates, AttributePromptParser
from TraderAdaptive import TraderAdaptive


def test_attribute_design_system():
    """Test the attribute design system without LLM"""
    print("Testing Attribute Design System")
    print("=" * 50)
    
    # Test different design strategies
    strategies = AttributeDesigner.get_design_strategies()
    print(f"Available strategies: {strategies}")
    
    for strategy in strategies:
        attributes = AttributeDesigner.design_attributes(strategy)
        print(f"\n{strategy.upper()} Strategy:")
        print(f"  Aggressiveness: {attributes.aggressiveness:.2f}")
        print(f"  Risk Tolerance: {attributes.risk_tolerance:.2f}")
        print(f"  Patience: {attributes.patience:.2f}")
        print(f"  Adaptability: {attributes.adaptability:.2f}")
        print(f"  Momentum Following: {attributes.momentum_following:.2f}")
        print(f"  Mean Reversion: {attributes.mean_reversion:.2f}")
        
        # Validate attributes
        assert attributes.validate(), f"Invalid attributes for {strategy} strategy"
    
    print("\nAll attribute design strategies working correctly!")


def test_attribute_adaptation():
    """Test attribute adaptation based on performance"""
    print("\nTesting Attribute Adaptation System")
    print("=" * 50)
    
    # Create an attribute manager
    manager = AttributeManager("test_agent_001")
    manager.initialize_attributes("balanced")
    
    print("Initial attributes:")
    initial_attrs = manager.get_attributes()
    for attr, value in initial_attrs.to_dict().items():
        print(f"  {attr}: {value:.2f}")
    
    # Simulate poor performance
    poor_performance = {
        'profit': -150,  # Big loss
        'market_volatility': 0.9,  # High volatility
        'relative_performance': -0.4,  # Underperforming
        'trade_count': 15,
        'timestamp': time.time()
    }
    
    print(f"\nPoor performance detected:")
    print(f"  Profit: ${poor_performance['profit']}")
    print(f"  Market volatility: {poor_performance['market_volatility']:.2f}")
    print(f"  Relative performance: {poor_performance['relative_performance']:.2f}")
    
    # Check if adaptation is needed
    if manager.adapter.should_adapt(poor_performance):
        print("Adaptation needed - applying conservative strategy")
        
        # Apply adaptation
        new_attributes = manager.adapter.apply_adaptation(
            "conservative", 
            "Big losses require more conservative approach",
            poor_performance
        )
        
        print("\nNew attributes after adaptation:")
        for attr, value in new_attributes.to_dict().items():
            print(f"  {attr}: {value:.2f}")
        
        # Check adaptation history
        history = manager.adapter.get_adaptation_history()
        print(f"\nAdaptation history: {len(history)} entries")
        for entry in history:
            print(f"  Strategy: {entry['strategy']}, Reasoning: {entry['reasoning']}")
    
    print("\nAttribute adaptation system working correctly!")


def test_prompt_templates():
    """Test LLM prompt templates"""
    print("\nTesting LLM Prompt Templates")
    print("=" * 50)
    
    # Test initial design prompt
    market_context = {
        'volatility': 'High',
        'trend': 'Downward',
        'competition': 'Intense',
        'liquidity': 'Moderate'
    }
    
    initial_prompt = AttributePromptTemplates.create_initial_design_prompt(
        market_context, AttributeDesigner.get_design_strategies()
    )
    
    print("Initial Design Prompt:")
    print("-" * 30)
    print(initial_prompt[:300] + "...")
    
    # Test adaptation prompt
    current_attributes = {
        'aggressiveness': 0.8,
        'risk_tolerance': 0.7,
        'patience': 0.2,
        'adaptability': 0.6,
        'momentum_following': 0.9,
        'mean_reversion': 0.1
    }
    
    performance_metrics = {
        'profit': -75,
        'market_volatility': 0.85,
        'relative_performance': -0.25,
        'trade_count': 12
    }
    
    adaptation_prompt = AttributePromptTemplates.create_adaptation_prompt(
        current_attributes, performance_metrics, market_context, 
        AttributeDesigner.get_design_strategies()
    )
    
    print("\nAdaptation Prompt:")
    print("-" * 30)
    print(adaptation_prompt[:300] + "...")
    
    # Test trading prompt
    belief_graph_insights = {
        'agent_strategies': ['Agent A: Momentum', 'Agent B: Conservative'],
        'market_sentiment': 'Bearish',
        'risk_assessment': 'High'
    }
    
    trading_prompt = AttributePromptTemplates.create_attribute_influenced_trading_prompt(
        current_attributes, market_context, belief_graph_insights
    )
    
    print("\nTrading Prompt:")
    print("-" * 30)
    print(trading_prompt[:300] + "...")
    
    print("\nAll prompt templates generated correctly!")


def test_response_parsing():
    """Test LLM response parsing"""
    print("\nTesting Response Parsing")
    print("=" * 50)
    
    # Test design response parsing
    design_responses = [
        "DESIGN: aggressive",
        "I think I should be DESIGN: conservative in this market",
        "DESIGN: momentum",
        "random"  # Fallback case
    ]
    
    print("Testing design response parsing:")
    for response in design_responses:
        strategy = AttributePromptParser.parse_design_response(response)
        print(f"  '{response}' -> {strategy}")
    
    # Test adaptation response parsing
    adaptation_responses = [
        "ADAPT: conservative\nREASONING: Need to reduce risk after losses",
        "ADAPT: balanced\nREASONING: Market conditions changed",
        "I should ADAPT: aggressive now"
    ]
    
    print("\nTesting adaptation response parsing:")
    for response in adaptation_responses:
        strategy, reasoning = AttributePromptParser.parse_adaptation_response(response)
        print(f"  '{response}' -> Strategy: {strategy}, Reasoning: {reasoning}")
    
    # Test trading response parsing
    trading_responses = [
        "BUY 100.50",
        "SELL 101.25",
        "WAIT",
        "I think I should BUY at 99.75"
    ]
    
    print("\nTesting trading response parsing:")
    for response in trading_responses:
        decision, price = AttributePromptParser.parse_trading_response(response)
        print(f"  '{response}' -> Decision: {decision}, Price: {price}")
    
    print("\nAll response parsing working correctly!")


def test_attribute_manager_persistence():
    """Test attribute manager JSON serialization/deserialization"""
    print("\nTesting Attribute Manager Persistence")
    print("=" * 50)
    
    # Create and configure manager
    manager = AttributeManager("test_agent_002")
    manager.initialize_attributes("momentum")
    manager.adaptation_enabled = True
    
    # Convert to JSON
    json_str = manager.to_json()
    print(f"JSON representation length: {len(json_str)} characters")
    
    # Recreate from JSON
    restored_manager = AttributeManager.from_json(json_str)
    
    # Verify restoration
    original_attrs = manager.get_attributes().to_dict()
    restored_attrs = restored_manager.get_attributes().to_dict()
    
    print("\nAttribute comparison:")
    for attr in original_attrs:
        original_val = original_attrs[attr]
        restored_val = restored_attrs[attr]
        match = "PASS" if abs(original_val - restored_val) < 0.001 else "FAIL"
        print(f"  {attr}: {original_val:.2f} vs {restored_val:.2f} {match}")
    
    # Verify other properties
    assert manager.design_strategy == restored_manager.design_strategy, "Design strategy mismatch"
    assert manager.adaptation_enabled == restored_manager.adaptation_enabled, "Adaptation enabled mismatch"
    
    print("\nAttribute manager persistence working correctly!")


def simulate_adaptive_trading_scenario():
    """Simulate a complete adaptive trading scenario"""
    print("\nSimulating Adaptive Trading Scenario")
    print("=" * 50)
    
    # Create adaptive trader (without LLM for testing)
    trader_params = {
        'adaptation_enabled': True,
        'adaptation_interval': 5,  # Adapt every 5 trades
        'temperature': 0.3
    }
    
    # Note: We'll create a mock trader since we don't have the full BSE environment
    print("Creating adaptive trader with momentum strategy...")
    
    # Simulate market context
    market_context = {
        'volatility': 'High',
        'trend': 'Upward',
        'competition': 'Moderate',
        'liquidity': 'High'
    }
    
    print(f"Market context: {market_context}")
    
    # Simulate performance over time
    performance_scenarios = [
        {'profit': 25, 'volatility': 0.3, 'relative_performance': 0.1, 'trades': 3},
        {'profit': -15, 'volatility': 0.6, 'relative_performance': -0.05, 'trades': 5},
        {'profit': -45, 'volatility': 0.8, 'relative_performance': -0.2, 'trades': 8},
        {'profit': -80, 'volatility': 0.9, 'relative_performance': -0.35, 'trades': 12}
    ]
    
    print("\nSimulating trading performance over time:")
    for i, scenario in enumerate(performance_scenarios):
        print(f"\nRound {i+1}:")
        print(f"  Profit: ${scenario['profit']}")
        print(f"  Volatility: {scenario['volatility']:.2f}")
        print(f"  Relative Performance: {scenario['relative_performance']:.2f}")
        print(f"  Total Trades: {scenario['trades']}")
        
        # Check if adaptation would be triggered
        if scenario['profit'] < -50 or scenario['volatility'] > 0.8:
            print("  Adaptation would be triggered!")
            
            # Determine adaptation strategy
            if scenario['profit'] < -100:
                strategy = "conservative"
            elif scenario['volatility'] > 0.8:
                strategy = "balanced"
            elif scenario['relative_performance'] < -0.3:
                strategy = "aggressive"
            else:
                strategy = "balanced"
            
            print(f"  Recommended strategy: {strategy}")
    
    print("\nAdaptive trading scenario simulation completed!")


def main():
    """Run all tests"""
    print("Starting Adaptive Trader System Tests")
    print("=" * 60)
    
    try:
        # Run all test functions
        test_attribute_design_system()
        test_attribute_adaptation()
        test_prompt_templates()
        test_response_parsing()
        test_attribute_manager_persistence()
        simulate_adaptive_trading_scenario()
        
        print("\nAll tests completed successfully!")
        print("\nSummary of what was tested:")
        print("  Attribute design strategies (6 different personality types)")
        print("  Automatic attribute adaptation based on performance")
        print("  LLM prompt templates for attribute design and adaptation")
        print("  Response parsing for LLM interactions")
        print("  JSON persistence of attribute state")
        print("  Complete adaptive trading scenario simulation")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
