#!/usr/bin/env python3
"""
Proprietary Trader Net Worth Analysis Script for BSE
Works with the CSV file created directly by BSE.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

def analyze_proprietary_traders(filename):
    """Analyze proprietary trader net worth changes over time from BSE.py CSV"""
    print(f"Analyzing proprietary traders in {filename}...")

    prop_traders = defaultdict(lambda: {
        'net_worths': [],
        'timestamps': []
    })

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        trader_columns = [col for col in reader.fieldnames if col.endswith('_NetWorth')]

        for row in reader:
            timestamp = int(row['Timestamp'].strip())

            for col in trader_columns:
                trader_type = col.replace('_NetWorth', '')
                net_worth = int(float(row[col].strip()))

                prop_traders[trader_type]['net_worths'].append(net_worth)
                prop_traders[trader_type]['timestamps'].append(timestamp)

    for trader_type in prop_traders.keys():
        if len(prop_traders[trader_type]['timestamps']) > 0:
            if prop_traders[trader_type]['timestamps'][0] != 0:
                prop_traders[trader_type]['timestamps'].insert(0, 0)
                prop_traders[trader_type]['net_worths'].insert(0, 500)

    return prop_traders

def print_final_net_worths(prop_traders):
    """Print final net worths of proprietary traders"""
    print("\n" + "="*60)
    print("FINAL PROPRIETARY TRADER NET WORTHS")
    print("="*60)
    
    final_results = []
    starting_balance = 500  # All proprietary traders start with $500
    
    for trader_type, data in prop_traders.items():
        if len(data['net_worths']) > 0:
            final_net_worth = data['net_worths'][-1]
            total_profit = final_net_worth - starting_balance
            
            final_results.append({
                'trader': trader_type,
                'net_worth': final_net_worth,
                'profit': total_profit,
                'trades': len(data['net_worths']) - 1  # Number of changes indicates trading activity
            })
    
    # Sort by net worth (descending)
    final_results.sort(key=lambda x: x['net_worth'], reverse=True)
    
    print(f"{'Rank':<4} {'Trader':<20} {'Net Worth':<10} {'Profit':<8} {'Changes':<8}")
    print("-" * 60)

    for rank, result in enumerate(final_results, 1):
        profit_str = f"+${result['profit']}" if result['profit'] >= 0 else f"-${abs(result['profit'])}"
        print(f"{rank:<4} {result['trader']:<20} ${result['net_worth']:<9} {profit_str:<8} {result['trades']:<8}")
    
    print("="*60)

def create_research_plot(prop_traders, output_filename):
    """Create a research-quality plot with bar chart and line chart"""
    print(f"Creating research plot: {output_filename}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    trader_names = list(prop_traders.keys())
    cmap = plt.colormaps.get_cmap('tab20')
    colors = {name: cmap(i % 20) for i, name in enumerate(trader_names)}
    
    # Plot 1: Bar chart of final net worths
    final_net_worths = []
    trader_names = []
    
    for trader_type, data in prop_traders.items():
        if len(data['net_worths']) > 0:
            final_net_worths.append(data['net_worths'][-1])
            trader_names.append(trader_type)
    
    sorted_data = sorted(zip(trader_names, final_net_worths), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_net_worths = zip(*sorted_data)

    bars = ax1.bar(range(len(sorted_names)), sorted_net_worths,
                   color=[colors[name] for name in sorted_names],
                   alpha=0.8, edgecolor='black', linewidth=1)

    for i, (bar, net_worth) in enumerate(zip(bars, sorted_net_worths)):
        height = bar.get_height()
        ax1.text(i, height + 5,
                f'${net_worth}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_title('Final Net Worth by Trader Type', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Net Worth ($)', fontsize=12)
    ax1.set_xticks(range(len(sorted_names)))
    ax1.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(sorted_net_worths) * 1.15)
    
    for trader_type, data in prop_traders.items():
        if len(data['net_worths']) > 0:
            ax2.plot(data['timestamps'], data['net_worths'],
                    label=trader_type, color=colors[trader_type],
                    linewidth=2, marker='o', markersize=4, alpha=0.7)

    ax2.set_title('Net Worth Evolution Over Time', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Net Worth ($)', fontsize=12)
    ax2.legend(fontsize=8, framealpha=0.9, ncol=3, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    starting_balance = 500
    ax2.axhline(y=starting_balance, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Research plot created: {output_filename}")

def main():
    """Main analysis function"""
    if len(sys.argv) > 1:
        net_worth_file = sys.argv[1]
    else:
        net_worth_file = "bse_d000_i10_0001_prop_net_worths.csv"
    
    try:
        # Analyze proprietary trader data
        prop_traders = analyze_proprietary_traders(net_worth_file)
        
        if not prop_traders:
            print("No proprietary trader data found!")
            return
        
        # Print final net worths
        print_final_net_worths(prop_traders)
        
        # Create research plot
        plot_filename = net_worth_file.replace('.csv', '_research_plot.png')
        create_research_plot(prop_traders, plot_filename)
        
        print(f"\nAnalysis complete!")
        print(f"Research plot: {plot_filename}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        print("Make sure you've run the BSE simulation first!")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
