#!/usr/bin/env python3
"""
LLM Attribute Analysis System

This module analyzes how LLM agents design and adapt their own trading attributes,
tracking patterns, performance correlations, and adaptation effectiveness.
"""

import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from agent_attributes import AttributeManager, AttributeDesigner
from llm_attribute_prompts import AttributePromptTemplates


class LLMAttributeAnalyzer:
    """
    Analyzes LLM agent attribute design and adaptation patterns
    """
    
    def __init__(self):
        self.agent_data = {}  # Store data for each agent
        self.adaptation_history = []  # Global adaptation history
        self.market_contexts = []  # Market context snapshots
        self.performance_correlations = {}  # Performance vs attribute correlations
        
    def add_agent_data(self, agent_id: str, attribute_data: Dict[str, Any]) -> None:
        """Add or update agent attribute data"""
        if agent_id not in self.agent_data:
            self.agent_data[agent_id] = {
                'initial_attributes': None,
                'current_attributes': None,
                'adaptation_history': [],
                'performance_history': [],
                'design_strategy': None,
                'creation_time': time.time()
            }
        
        # Update current attributes
        if 'attributes' in attribute_data:
            self.agent_data[agent_id]['current_attributes'] = attribute_data['attributes']
        
        # Update design strategy
        if 'design_strategy' in attribute_data:
            self.agent_data[agent_id]['design_strategy'] = attribute_data['design_strategy']
        
        # Add adaptation history
        if 'adaptation_history' in attribute_data:
            self.agent_data[agent_id]['adaptation_history'].extend(attribute_data['adaptation_history'])
            self.adaptation_history.extend(attribute_data['adaptation_history'])
        
        # Add performance metrics
        if 'performance_metrics' in attribute_data:
            self.agent_data[agent_id]['performance_history'].append(attribute_data['performance_metrics'])
    
    def add_market_context(self, context: Dict[str, Any]) -> None:
        """Add market context snapshot"""
        context['timestamp'] = time.time()
        self.market_contexts.append(context)
    
    def analyze_attribute_distributions(self) -> Dict[str, Any]:
        """Analyze the distribution of attribute values across all agents"""
        print("Analyzing Attribute Distributions...")
        
        all_attributes = []
        for agent_id, data in self.agent_data.items():
            if data['current_attributes']:
                all_attributes.append(data['current_attributes'])
        
        if not all_attributes:
            return {"error": "No attribute data available"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_attributes)
        
        analysis = {
            'total_agents': len(all_attributes),
            'attribute_means': df.mean().to_dict(),
            'attribute_stds': df.std().to_dict(),
            'attribute_ranges': {
                col: {'min': df[col].min(), 'max': df[col].max()} 
                for col in df.columns
            },
            'correlation_matrix': df.corr().to_dict()
        }
        
        print(f"Analyzed {len(all_attributes)} agents")
        return analysis
    
    def analyze_design_strategies(self) -> Dict[str, Any]:
        """Analyze which design strategies agents choose"""
        print("Analyzing Design Strategy Choices...")
        
        strategy_counts = Counter()
        strategy_performance = defaultdict(list)
        
        for agent_id, data in self.agent_data.items():
            strategy = data.get('design_strategy', 'unknown')
            strategy_counts[strategy] += 1
            
            # Calculate average performance for each strategy
            if data['performance_history']:
                avg_profit = sum(p.get('profit', 0) for p in data['performance_history']) / len(data['performance_history'])
                strategy_performance[strategy].append(avg_profit)
        
        # Calculate average performance per strategy
        strategy_avg_performance = {}
        for strategy, profits in strategy_performance.items():
            if profits:
                strategy_avg_performance[strategy] = sum(profits) / len(profits)
        
        analysis = {
            'strategy_distribution': dict(strategy_counts),
            'strategy_performance': strategy_avg_performance,
            'most_popular_strategy': strategy_counts.most_common(1)[0] if strategy_counts else None,
            'best_performing_strategy': max(strategy_avg_performance.items(), key=lambda x: x[1]) if strategy_avg_performance else None
        }
        
        print(f"Found {len(strategy_counts)} different strategies")
        return analysis
    
    def analyze_adaptation_patterns(self) -> Dict[str, Any]:
        """Analyze how and when agents adapt their attributes"""
        print("Analyzing Adaptation Patterns...")
        
        if not self.adaptation_history:
            return {"error": "No adaptation history available"}
        
        # Convert to DataFrame
        df = pd.DataFrame(self.adaptation_history)
        
        # Analyze adaptation triggers
        trigger_analysis = {}
        if 'performance' in df.columns:
            # Analyze what performance metrics trigger adaptations
            for _, row in df.iterrows():
                performance = row.get('performance', {})
                if isinstance(performance, dict):
                    for metric, value in performance.items():
                        if metric not in trigger_analysis:
                            trigger_analysis[metric] = []
                        trigger_analysis[metric].append(value)
        
        # Analyze adaptation strategies
        strategy_counts = Counter(df.get('strategy', []))
        
        # Analyze adaptation frequency over time
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['hour'] = df['timestamp'].dt.hour
            hourly_adaptations = df.groupby('hour').size()
        else:
            hourly_adaptations = {}
        
        analysis = {
            'total_adaptations': len(self.adaptation_history),
            'adaptation_triggers': trigger_analysis,
            'strategy_changes': dict(strategy_counts),
            'hourly_pattern': hourly_adaptations.to_dict() if hasattr(hourly_adaptations, 'to_dict') else hourly_adaptations,
            'most_common_adaptation': strategy_counts.most_common(1)[0] if strategy_counts else None
        }
        
        print(f"Analyzed {len(self.adaptation_history)} adaptations")
        return analysis
    
    def analyze_performance_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between attributes and performance"""
        print("Analyzing Performance Correlations...")
        
        correlations = {}
        
        for agent_id, data in self.agent_data.items():
            if not data['current_attributes'] or not data['performance_history']:
                continue
            
            attributes = data['current_attributes']
            avg_profit = sum(p.get('profit', 0) for p in data['performance_history']) / len(data['performance_history'])
            
            # Calculate correlation for each attribute
            for attr, value in attributes.items():
                if attr not in correlations:
                    correlations[attr] = []
                correlations[attr].append((value, avg_profit))
        
        # Calculate correlation coefficients
        correlation_coefficients = {}
        for attr, data_points in correlations.items():
            if len(data_points) > 1:
                # Simple correlation calculation
                x_values = [point[0] for point in data_points]
                y_values = [point[1] for point in data_points]
                
                # Calculate Pearson correlation
                correlation = self._calculate_correlation(x_values, y_values)
                correlation_coefficients[attr] = correlation
        
        # Sort by absolute correlation strength
        sorted_correlations = sorted(
            correlation_coefficients.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        analysis = {
            'correlation_coefficients': dict(sorted_correlations),
            'strongest_correlations': sorted_correlations[:3],
            'total_agents_analyzed': len([d for d in self.agent_data.values() if d['performance_history']])
        }
        
        print(f"Analyzed correlations for {len(correlation_coefficients)} attributes")
        return analysis
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def analyze_agent_evolution(self, agent_id: str) -> Dict[str, Any]:
        """Analyze how a specific agent's attributes evolved over time"""
        print(f"Analyzing Evolution for Agent {agent_id}...")
        
        if agent_id not in self.agent_data:
            return {"error": f"Agent {agent_id} not found"}
        
        data = self.agent_data[agent_id]
        
        if not data['adaptation_history']:
            return {"error": "No adaptation history for this agent"}
        
        # Sort adaptations by timestamp
        adaptations = sorted(data['adaptation_history'], key=lambda x: x.get('timestamp', 0))
        
        evolution = {
            'agent_id': agent_id,
            'initial_strategy': data.get('design_strategy'),
            'total_adaptations': len(adaptations),
            'adaptation_timeline': [],
            'strategy_changes': []
        }
        
        for i, adaptation in enumerate(adaptations):
            evolution['adaptation_timeline'].append({
                'step': i + 1,
                'timestamp': adaptation.get('timestamp', 0),
                'old_strategy': adaptation.get('old_attributes', {}),
                'new_strategy': adaptation.get('new_attributes', {}),
                'reasoning': adaptation.get('reasoning', ''),
                'performance': adaptation.get('performance', {})
            })
            
            if 'strategy' in adaptation:
                evolution['strategy_changes'].append(adaptation['strategy'])
        
        print(f"Analyzed {len(adaptations)} adaptations for agent {agent_id}")
        return evolution
    
    def generate_visualizations(self, output_dir: str = ".") -> Dict[str, str]:
        """Generate visualizations of the analysis results"""
        print("Generating Visualizations...")
        
        output_files = {}
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Attribute Distribution Heatmap
            if self.agent_data:
                self._create_attribute_heatmap(output_dir)
                output_files['attribute_heatmap'] = f"{output_dir}/attribute_distribution_heatmap.png"
            
            # 2. Design Strategy Distribution
            if self.agent_data:
                self._create_strategy_distribution(output_dir)
                output_files['strategy_distribution'] = f"{output_dir}/design_strategy_distribution.png"
            
            # 3. Adaptation Timeline
            if self.adaptation_history:
                self._create_adaptation_timeline(output_dir)
                output_files['adaptation_timeline'] = f"{output_dir}/adaptation_timeline.png"
            
            # 4. Performance Correlation Scatter
            if self.agent_data:
                self._create_performance_correlation_scatter(output_dir)
                output_files['performance_correlation'] = f"{output_dir}/performance_correlation_scatter.png"
            
            print(f"Generated {len(output_files)} visualizations")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
        
        return output_files
    
    def _create_attribute_heatmap(self, output_dir: str) -> None:
        """Create heatmap of attribute distributions"""
        all_attributes = []
        for data in self.agent_data.values():
            if data['current_attributes']:
                all_attributes.append(data['current_attributes'])
        
        if not all_attributes:
            return
        
        df = pd.DataFrame(all_attributes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Attribute Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/attribute_distribution_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_strategy_distribution(self, output_dir: str) -> None:
        """Create bar chart of design strategy distribution"""
        strategy_counts = Counter()
        for data in self.agent_data.values():
            strategy = data.get('design_strategy', 'unknown')
            strategy_counts[strategy] += 1
        
        if not strategy_counts:
            return
        
        plt.figure(figsize=(12, 6))
        strategies, counts = zip(*strategy_counts.most_common())
        bars = plt.bar(strategies, counts, color=sns.color_palette("husl", len(strategies)))
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        plt.title('Distribution of Agent Design Strategies')
        plt.xlabel('Design Strategy')
        plt.ylabel('Number of Agents')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/design_strategy_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_adaptation_timeline(self, output_dir: str) -> None:
        """Create timeline of adaptations"""
        if not self.adaptation_history:
            return
        
        df = pd.DataFrame(self.adaptation_history)
        if 'timestamp' not in df.columns:
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['timestamp'].dt.hour
        
        hourly_counts = df.groupby('hour').size()
        
        plt.figure(figsize=(12, 6))
        plt.plot(hourly_counts.index, hourly_counts.values, marker='o', linewidth=2, markersize=8)
        plt.title('Adaptation Frequency by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Adaptations')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/adaptation_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_correlation_scatter(self, output_dir: str) -> None:
        """Create scatter plots of attributes vs performance"""
        # Get correlation data
        correlations = self.analyze_performance_correlations()
        if 'correlation_coefficients' not in correlations:
            return
        
        # Create subplots for top 4 attributes
        top_attrs = list(correlations['correlation_coefficients'].keys())[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, attr in enumerate(top_attrs):
            if i >= 4:
                break
            
            # Collect data points
            x_values, y_values = [], []
            for agent_id, data in self.agent_data.items():
                if data['current_attributes'] and data['performance_history']:
                    attr_value = data['current_attributes'].get(attr, 0)
                    avg_profit = sum(p.get('profit', 0) for p in data['performance_history']) / len(data['performance_history'])
                    x_values.append(attr_value)
                    y_values.append(avg_profit)
            
            if x_values and y_values:
                axes[i].scatter(x_values, y_values, alpha=0.6, s=50)
                axes[i].set_xlabel(attr.replace('_', ' ').title())
                axes[i].set_ylabel('Average Profit')
                axes[i].set_title(f'{attr.replace("_", " ").title()} vs Performance')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_correlation_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_file: str = "llm_attribute_analysis_report.json") -> str:
        """Generate comprehensive analysis report"""
        print("Generating Analysis Report...")
        
        report = {
            'timestamp': time.time(),
            'summary': {
                'total_agents': len(self.agent_data),
                'total_adaptations': len(self.adaptation_history),
                'analysis_period': self._get_analysis_period()
            },
            'attribute_distributions': self.analyze_attribute_distributions(),
            'design_strategies': self.analyze_design_strategies(),
            'adaptation_patterns': self.analyze_adaptation_patterns(),
            'performance_correlations': self.analyze_performance_correlations(),
            'agent_evolutions': {}
        }
        
        # Add individual agent evolution analysis
        for agent_id in self.agent_data.keys():
            report['agent_evolutions'][agent_id] = self.analyze_agent_evolution(agent_id)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to {output_file}")
        return output_file
    
    def _get_analysis_period(self) -> Dict[str, Any]:
        """Calculate the time period covered by the analysis"""
        if not self.agent_data:
            return {"start": None, "end": None, "duration_hours": 0}
        
        start_time = min(data.get('creation_time', time.time()) for data in self.agent_data.values())
        end_time = max(data.get('creation_time', time.time()) for data in self.agent_data.values())
        
        duration_hours = (end_time - start_time) / 3600
        
        return {
            "start": start_time,
            "end": end_time,
            "duration_hours": duration_hours
        }
    
    def print_summary(self) -> None:
        """Print a summary of the analysis"""
        print("\n" + "="*60)
        print("LLM ATTRIBUTE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total Agents Analyzed: {len(self.agent_data)}")
        print(f"Total Adaptations: {len(self.adaptation_history)}")
        print(f"Market Context Snapshots: {len(self.market_contexts)}")
        
        if self.agent_data:
            # Most common design strategy
            strategies = [data.get('design_strategy', 'unknown') for data in self.agent_data.values()]
            strategy_counts = Counter(strategies)
            most_common = strategy_counts.most_common(1)[0] if strategy_counts else None
            
            if most_common:
                print(f"Most Popular Design Strategy: {most_common[0]} ({most_common[1]} agents)")
            
            # Adaptation rate
            agents_with_adaptations = sum(1 for data in self.agent_data.values() if data['adaptation_history'])
            adaptation_rate = (agents_with_adaptations / len(self.agent_data)) * 100
            print(f"Agents That Adapted: {agents_with_adaptations} ({adaptation_rate:.1f}%)")
        
        print("="*60)


def main():
    """Demo the LLM attribute analyzer"""
    print("LLM Attribute Analyzer Demo")
    print("="*50)
    
    # Create analyzer
    analyzer = LLMAttributeAnalyzer()
    
    # Simulate some agent data
    print("Simulating agent data...")
    
    # Agent 1: Conservative trader that adapts to aggressive
    agent1_data = {
        'attributes': {
            'aggressiveness': 0.2,
            'risk_tolerance': 0.3,
            'patience': 0.8,
            'adaptability': 0.6,
            'momentum_following': 0.3,
            'mean_reversion': 0.7
        },
        'design_strategy': 'conservative',
        'adaptation_history': [
            {
                'timestamp': time.time() - 3600,
                'old_attributes': {'aggressiveness': 0.2, 'risk_tolerance': 0.3},
                'new_attributes': {'aggressiveness': 0.7, 'risk_tolerance': 0.6},
                'strategy': 'aggressive',
                'reasoning': 'Market conditions changed, need to be more active',
                'performance': {'profit': -50, 'market_volatility': 0.8}
            }
        ],
        'performance_metrics': {
            'profit': 25,
            'market_volatility': 0.4,
            'relative_performance': 0.1,
            'trade_count': 8
        }
    }
    
    # Agent 2: Momentum trader that stays consistent
    agent2_data = {
        'attributes': {
            'aggressiveness': 0.7,
            'risk_tolerance': 0.6,
            'patience': 0.3,
            'adaptability': 0.2,
            'momentum_following': 0.9,
            'mean_reversion': 0.1
        },
        'design_strategy': 'momentum',
        'adaptation_history': [],
        'performance_metrics': {
            'profit': 45,
            'market_volatility': 0.3,
            'relative_performance': 0.2,
            'trade_count': 12
        }
    }
    
    # Add agents to analyzer
    analyzer.add_agent_data('agent_001', agent1_data)
    analyzer.add_agent_data('agent_002', agent2_data)
    
    # Add market context
    analyzer.add_market_context({
        'volatility': 'High',
        'trend': 'Upward',
        'competition': 'Moderate'
    })
    
    # Run analysis
    print("\nRunning analysis...")
    
    # Generate report
    report_file = analyzer.generate_report()
    
    # Print summary
    analyzer.print_summary()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    viz_files = analyzer.generate_visualizations()
    
    print(f"\nAnalysis complete! Report saved to: {report_file}")
    print(f"Visualizations generated: {len(viz_files)} files")
    
    # Print some key insights
    print("\nKey Insights:")
    
    # Attribute distributions
    attr_analysis = analyzer.analyze_attribute_distributions()
    if 'attribute_means' in attr_analysis:
        print(f"  • Average aggressiveness across agents: {attr_analysis['attribute_means'].get('aggressiveness', 0):.2f}")
        print(f"  • Average patience across agents: {attr_analysis['attribute_means'].get('patience', 0):.2f}")
    
    # Design strategies
    strategy_analysis = analyzer.analyze_design_strategies()
    if 'most_popular_strategy' in strategy_analysis and strategy_analysis['most_popular_strategy']:
        strategy, count = strategy_analysis['most_popular_strategy']
        print(f"  • Most popular design strategy: {strategy} ({count} agents)")
    
    # Performance correlations
    perf_analysis = analyzer.analyze_performance_correlations()
    if 'strongest_correlations' in perf_analysis and perf_analysis['strongest_correlations']:
        strongest_attr, correlation = perf_analysis['strongest_correlations'][0]
        print(f"  • Strongest performance correlation: {strongest_attr} (r={correlation:.3f})")


if __name__ == "__main__":
    main()