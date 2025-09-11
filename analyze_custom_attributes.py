#!/usr/bin/env python3
"""
Quantitative Analysis of LLM-Designed Custom Attributes

This script analyzes the quantitative results of the custom attributes system
for presentation to mentors and stakeholders.
"""

import json
import statistics
from typing import Dict, List, Any
from llm_designed_attributes import CustomAttributeManager

def analyze_custom_attributes():
    """Analyze custom attributes quantitatively"""
    
    print("QUANTITATIVE ANALYSIS: LLM-Designed Custom Attributes")
    print("=" * 70)
    
    # Test different market conditions
    market_conditions = [
        {
            'name': 'Low Volatility, Sideways',
            'context': {'volatility': 'Low', 'trend': 'Sideways', 'competition': 'Low', 'liquidity': 'High'}
        },
        {
            'name': 'High Volatility, Upward',
            'context': {'volatility': 'High', 'trend': 'Upward', 'competition': 'Moderate', 'liquidity': 'High'}
        },
        {
            'name': 'Extreme Volatility, Strong Upward',
            'context': {'volatility': 'Extreme', 'trend': 'Strong Upward', 'competition': 'Intense', 'liquidity': 'Very High'}
        }
    ]
    
    all_attributes = []
    attribute_names = set()
    market_results = []
    
    for i, condition in enumerate(market_conditions, 1):
        print(f"\nMARKET CONDITION {i}: {condition['name']}")
        print("-" * 50)
        
        try:
            manager = CustomAttributeManager(f"ANALYSIS_{i}")
            attributes = manager.initialize_custom_attributes(condition['context'])
            
            # Collect quantitative data
            condition_data = {
                'market_condition': condition['name'],
                'attribute_count': len(attributes.attributes),
                'attributes': [],
                'average_value': 0.0,
                'value_range': [0.0, 0.0]
            }
            
            values = []
            for attr in attributes.attributes:
                attribute_names.add(attr.name)
                all_attributes.append({
                    'name': attr.name,
                    'value': attr.value,
                    'market_condition': condition['name']
                })
                
                condition_data['attributes'].append({
                    'name': attr.name,
                    'value': attr.value,
                    'description': attr.description
                })
                
                values.append(attr.value)
                
                print(f"  • {attr.name}: {attr.value:.2f}")
            
            # Calculate statistics
            if values:
                condition_data['average_value'] = statistics.mean(values)
                condition_data['value_range'] = [min(values), max(values)]
                condition_data['standard_deviation'] = statistics.stdev(values) if len(values) > 1 else 0.0
            
            market_results.append(condition_data)
            
            print(f"  Average Attribute Value: {condition_data['average_value']:.2f}")
            print(f"  Value Range: {condition_data['value_range'][0]:.2f} - {condition_data['value_range'][1]:.2f}")
            print(f"  Standard Deviation: {condition_data['standard_deviation']:.2f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Overall quantitative analysis
    print(f"\nOVERALL QUANTITATIVE RESULTS")
    print("=" * 50)
    
    print(f"Total Market Conditions Tested: {len(market_conditions)}")
    print(f"Total Custom Attributes Created: {len(all_attributes)}")
    print(f"Unique Attribute Names: {len(attribute_names)}")
    print(f"Average Attributes per Condition: {len(all_attributes) / len(market_conditions):.1f}")
    
    # Attribute value analysis
    all_values = [attr['value'] for attr in all_attributes]
    if all_values:
        print(f"Overall Average Attribute Value: {statistics.mean(all_values):.2f}")
        print(f"Overall Value Range: {min(all_values):.2f} - {max(all_values):.2f}")
        print(f"Overall Standard Deviation: {statistics.stdev(all_values):.2f}")
    
    # Attribute frequency analysis
    print(f"\nATTRIBUTE FREQUENCY ANALYSIS")
    print("-" * 30)
    attribute_freq = {}
    for attr in all_attributes:
        name = attr['name']
        if name not in attribute_freq:
            attribute_freq[name] = 0
        attribute_freq[name] += 1
    
    for name, count in sorted(attribute_freq.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {count} occurrences")
    
    # Market condition comparison
    print(f"\nMARKET CONDITION COMPARISON")
    print("-" * 30)
    for result in market_results:
        print(f"  {result['market_condition']}:")
        print(f"    Attributes: {result['attribute_count']}")
        print(f"    Average Value: {result['average_value']:.2f}")
        print(f"    Value Range: {result['value_range'][0]:.2f} - {result['value_range'][1]:.2f}")
    
    # Save results to JSON
    analysis_results = {
        'summary': {
            'total_market_conditions': len(market_conditions),
            'total_attributes': len(all_attributes),
            'unique_attribute_names': len(attribute_names),
            'average_attributes_per_condition': len(all_attributes) / len(market_conditions),
            'overall_average_value': statistics.mean(all_values) if all_values else 0.0,
            'overall_value_range': [min(all_values), max(all_values)] if all_values else [0.0, 0.0],
            'overall_standard_deviation': statistics.stdev(all_values) if len(all_values) > 1 else 0.0
        },
        'market_conditions': market_results,
        'attribute_frequency': attribute_freq,
        'all_attributes': all_attributes
    }
    
    with open('custom_attributes_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nAnalysis saved to: custom_attributes_analysis.json")
    
    return analysis_results

if __name__ == "__main__":
    analyze_custom_attributes()
