#!/usr/bin/env python3
"""
Simplified Performance Analysis Script for BSE Trading Agents
Focuses on key metrics: Total Return, Average Return, Sharpe Ratio, Win Rate
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Define starting balances for different trader types
STARTING_BALANCES = {
    'GVWY': 0,    # Giveaway traders start with $0
    'SHVR': 0,    # Shaver traders start with $0
    'ZIC': 0,     # Zero Intelligence Constrained start with $0
    'ZIP': 0,     # Zero Intelligence Plus start with $0
    'LLM': 0,     # LLM traders start with $0
    'LLM_BG': 0,  # LLM with Belief Graph start with $0
    'PT1': 500,   # Proprietary traders start with $500
    'PT2': 500,   # Proprietary traders start with $500
}

def analyze_avg_balance(filename):
    """Analyze the average balance CSV file to extract performance metrics"""
    print(f"Analyzing {filename}...")
    
    # Data structure to store performance by agent type
    performance = defaultdict(lambda: {
        'balances': [],
        'total_profits': [],
        'trade_counts': [],
        'timestamps': [],
        'returns': []
    })
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
                
            # Parse the row - format: session_id, time, best_bid, best_ask, agent_data...
            session_id = row[0].strip()
            time = int(row[1].strip())
            best_bid = row[2].strip()
            best_ask = row[3].strip()
            
            # Parse agent data (groups of 4 columns: type, total_profit, count, avg_profit)
            i = 4
            while i < len(row) - 1:
                if i + 3 < len(row):
                    agent_type = row[i].strip()
                    total_profit = int(row[i+1].strip())
                    count = int(row[i+2].strip())
                    avg_profit = float(row[i+3].strip())
                    
                    # Store raw data
                    performance[agent_type]['balances'].append(total_profit)
                    performance[agent_type]['total_profits'].append(total_profit)
                    performance[agent_type]['trade_counts'].append(count)
                    performance[agent_type]['timestamps'].append(time)
                    
                    i += 4
                else:
                    break
    
    # Calculate proper returns
    for agent_type, data in performance.items():
        if len(data['balances']) > 0:
            starting_balance = STARTING_BALANCES.get(agent_type, 0)
            
            # Calculate total wealth (starting balance + profits)
            total_wealth = [starting_balance + profit for profit in data['total_profits']]
            
            # Calculate percentage returns (proper way for Sharpe ratio)
            percentage_returns = []
            for i, wealth in enumerate(total_wealth):
                if i == 0:
                    # First period: percentage return from starting balance
                    if starting_balance > 0:
                        pct_return = (wealth - starting_balance) / starting_balance
                    else:
                        pct_return = wealth  # For agents starting with $0
                else:
                    # Subsequent periods: percentage return from previous wealth
                    prev_wealth = total_wealth[i-1]
                    if prev_wealth > 0:
                        pct_return = (wealth - prev_wealth) / prev_wealth
                    else:
                        pct_return = 0
                
                percentage_returns.append(pct_return)
            
            # Calculate absolute returns (for other metrics)
            absolute_returns = []
            for i, wealth in enumerate(total_wealth):
                if i == 0:
                    # First period: return is change from starting balance
                    return_val = wealth - starting_balance
                else:
                    # Subsequent periods: return is change from previous wealth
                    prev_wealth = total_wealth[i-1]
                    return_val = wealth - prev_wealth
                
                absolute_returns.append(return_val)
            
            data['total_wealth'] = total_wealth
            data['returns'] = absolute_returns
            data['percentage_returns'] = percentage_returns
            data['starting_balance'] = starting_balance
    
    return performance

def print_performance_summary(performance):
    """Print a focused summary of key performance metrics"""
    print("\n" + "="*80)
    print("KEY PERFORMANCE METRICS SUMMARY")
    print("="*80)
    
    # Calculate key metrics for each agent
    agent_metrics = []
    
    for agent_type, data in performance.items():
        if len(data['balances']) > 0:
            starting_balance = data['starting_balance']
            final_wealth = data['total_wealth'][-1]
            total_return = final_wealth - starting_balance
            
            # Calculate average return per period
            avg_return = np.mean(data['returns']) if data['returns'] else 0
            
            # Calculate Sharpe ratio using percentage returns
            pct_returns_array = np.array(data['percentage_returns'])
            avg_pct_return = np.mean(pct_returns_array) if len(pct_returns_array) > 0 else 0
            pct_volatility = np.std(pct_returns_array) if len(pct_returns_array) > 1 else 0
            sharpe_ratio = avg_pct_return / pct_volatility if pct_volatility > 0 else 0
            
            # Calculate win rate
            positive_returns = pct_returns_array[pct_returns_array > 0]
            win_rate = len(positive_returns) / len(pct_returns_array) if len(pct_returns_array) > 0 else 0
            
            # Total trades
            total_trades = sum(data['trade_counts'])
            
            agent_metrics.append({
                'type': agent_type,
                'total_return': total_return,
                'avg_return': avg_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'final_wealth': final_wealth
            })
    
    # Sort by total return (descending)
    agent_metrics.sort(key=lambda x: x['total_return'], reverse=True)
    
    print(f"\n{'Rank':<4} {'Agent':<8} {'Total Return':<12} {'Avg Return':<10} {'Sharpe':<8} {'Win Rate':<10} {'Trades':<8}")
    print("-" * 80)
    
    for rank, agent in enumerate(agent_metrics, 1):
        print(f"{rank:<4} {agent['type']:<8} ${agent['total_return']:<11,} ${agent['avg_return']:<9.2f} {agent['sharpe_ratio']:<7.3f} {agent['win_rate']:<9.1%} {agent['total_trades']:<8,}")

def plot_key_metrics(performance):
    """Create focused visualizations of key performance metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Calculate metrics using the same logic as the summary
    agent_metrics = []
    
    for agent_type, data in performance.items():
        if len(data['balances']) > 0:
            starting_balance = data['starting_balance']
            final_wealth = data['total_wealth'][-1]
            total_return = final_wealth - starting_balance
            
            # Calculate average return per period
            avg_return = np.mean(data['returns']) if data['returns'] else 0
            
            # Calculate Sharpe ratio using percentage returns
            pct_returns_array = np.array(data['percentage_returns'])
            avg_pct_return = np.mean(pct_returns_array) if len(pct_returns_array) > 0 else 0
            pct_volatility = np.std(pct_returns_array) if len(pct_returns_array) > 1 else 0
            sharpe_ratio = avg_pct_return / pct_volatility if pct_volatility > 0 else 0
            
            # Calculate win rate
            positive_returns = pct_returns_array[pct_returns_array > 0]
            win_rate = len(positive_returns) / len(pct_returns_array) if len(pct_returns_array) > 0 else 0
            
            agent_metrics.append({
                'type': agent_type,
                'total_return': total_return,
                'avg_return': avg_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate
            })
    
    # Sort by total return (descending) to match summary
    agent_metrics.sort(key=lambda x: x['total_return'], reverse=True)
    
    # Extract data for plotting
    agent_types = [agent['type'] for agent in agent_metrics]
    total_returns = [agent['total_return'] for agent in agent_metrics]
    avg_returns = [agent['avg_return'] for agent in agent_metrics]
    sharpe_ratios = [agent['sharpe_ratio'] for agent in agent_metrics]
    win_rates = [agent['win_rate'] for agent in agent_metrics]
    

    
    # Plot 1: Total Return
    colors = ['green' if r >= 0 else 'red' for r in total_returns]
    bars1 = ax1.bar(agent_types, total_returns, color=colors, alpha=0.7)
    ax1.set_title('Total Return by Agent Type', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total Return ($)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, ret in zip(bars1, total_returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(total_returns)*0.01 if ret >= 0 else height - max(abs(r) for r in total_returns)*0.01,
                f'${ret:,}', ha='center', va='bottom' if ret >= 0 else 'top', fontsize=9)
    
    # Plot 2: Average Return per Period
    colors = ['green' if r >= 0 else 'red' for r in avg_returns]
    bars2 = ax2.bar(agent_types, avg_returns, color=colors, alpha=0.7)
    ax2.set_title('Average Return per Period', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Return ($)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, ret in zip(bars2, avg_returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(avg_returns)*0.01 if ret >= 0 else height - max(abs(r) for r in avg_returns)*0.01,
                f'${ret:.1f}', ha='center', va='bottom' if ret >= 0 else 'top', fontsize=9)
    
    # Plot 3: Sharpe Ratio
    colors = ['green' if s > 0 else 'red' for s in sharpe_ratios]
    bars3 = ax3.bar(agent_types, sharpe_ratios, color=colors, alpha=0.7)
    ax3.set_title('Sharpe Ratio (Risk-Adjusted Returns)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, sharpe in zip(bars3, sharpe_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(sharpe_ratios)*0.01 if sharpe >= 0 else height - max(abs(s) for s in sharpe_ratios)*0.01,
                f'{sharpe:.3f}', ha='center', va='bottom' if sharpe >= 0 else 'top', fontsize=9)
    
    # Plot 4: Win Rate
    colors = ['green' if w > 0.5 else 'orange' if w > 0.2 else 'red' for w in win_rates]
    bars4 = ax4.bar(agent_types, win_rates, color=colors, alpha=0.7)
    ax4.set_title('Win Rate (Percentage of Profitable Periods)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Win Rate')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 1)
    
    # Add value labels
    for bar, rate in zip(bars4, win_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('key_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_tape(filename):
    """Analyze the transaction tape to understand trading activity"""
    print(f"Analyzing {filename}...")
    
    transactions = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                # Format: TRD, time, price
                trans_type = row[0].strip()
                time = float(row[1].strip())
                price = int(row[2].strip())
                transactions.append({'type': trans_type, 'time': time, 'price': price})
    
    return transactions

def main():
    """Main analysis function"""
    # Analyze the files from our simulation
    balance_file = "bse_d000_i10_0001_avg_balance.csv"
    tape_file = "bse_d000_i10_0001_tape.csv"
    
    try:
        # Analyze performance data
        performance = analyze_avg_balance(balance_file)
        
        # Print focused summary
        print_performance_summary(performance)
        
        # Create focused visualizations
        plot_key_metrics(performance)
        
        # Analyze transaction data
        transactions = analyze_tape(tape_file)
        print(f"\nTotal Transactions: {len(transactions):,}")
        
        if transactions:
            prices = [t['price'] for t in transactions]
            print(f"Price Range: ${min(prices):,} - ${max(prices):,}")
            print(f"Average Price: ${np.mean(prices):,.2f}")
            print(f"Price Volatility: ${np.std(prices):,.2f}")
        
        print(f"\nAnalysis complete! Key metrics visualization saved as 'key_performance_metrics.png'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        print("Make sure you've run the BSE simulation first!")
        print("\nTo run a simulation:")
        print("  python BSE.py")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
