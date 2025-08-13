#!/usr/bin/env python3
"""
Performance Analysis Script for BSE Trading Agents
Analyzes the output files from BSE simulation to evaluate agent performance
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_avg_balance(filename):
    """Analyze the average balance CSV file to extract performance metrics"""
    print(f"Analyzing {filename}...")
    
    # Data structure to store performance by agent type
    performance = defaultdict(lambda: {
        'balances': [],
        'profits': [],
        'trade_counts': [],
        'avg_profit_per_trade': []
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
                    
                    performance[agent_type]['balances'].append(total_profit)
                    performance[agent_type]['profits'].append(avg_profit)
                    performance[agent_type]['trade_counts'].append(count)
                    
                    if avg_profit > 0 and count > 0:
                        performance[agent_type]['avg_profit_per_trade'].append(avg_profit)
                    
                    i += 4
                else:
                    break
    
    return performance

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

def print_performance_summary(performance):
    """Print a summary of agent performance"""
    print("\n" + "="*60)
    print("TRADING AGENT PERFORMANCE SUMMARY")
    print("="*60)
    
    for agent_type, data in performance.items():
        if len(data['balances']) > 0:
            final_balance = data['balances'][-1]
            max_balance = max(data['balances'])
            min_balance = min(data['balances'])
            avg_profit = np.mean(data['profits']) if data['profits'] else 0
            total_trades = sum(data['trade_counts'])
            
            print(f"\n{agent_type}:")
            print(f"  Final Balance: {final_balance:,}")
            print(f"  Max Balance: {max_balance:,}")
            print(f"  Min Balance: {min_balance:,}")
            print(f"  Average Profit per Agent: {avg_profit:.2f}")
            print(f"  Total Trades: {total_trades:,}")
            
            if data['avg_profit_per_trade']:
                avg_profit_per_trade = np.mean(data['avg_profit_per_trade'])
                print(f"  Average Profit per Trade: {avg_profit_per_trade:.2f}")

def plot_performance(performance):
    """Create performance visualization plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Final Balances
    agent_types = list(performance.keys())
    final_balances = [performance[agent]['balances'][-1] if performance[agent]['balances'] else 0 
                     for agent in agent_types]
    
    ax1.bar(agent_types, final_balances, color='skyblue')
    ax1.set_title('Final Balance by Agent Type')
    ax1.set_ylabel('Final Balance')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Average Profit per Agent
    avg_profits = [np.mean(performance[agent]['profits']) if performance[agent]['profits'] else 0 
                  for agent in agent_types]
    
    ax2.bar(agent_types, avg_profits, color='lightgreen')
    ax2.set_title('Average Profit per Agent')
    ax2.set_ylabel('Average Profit')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Balance Evolution Over Time
    for agent_type, data in performance.items():
        if data['balances']:
            ax3.plot(data['balances'], label=agent_type, linewidth=2)
    
    ax3.set_title('Balance Evolution Over Time')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Total Balance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trading Activity
    total_trades = [sum(performance[agent]['trade_counts']) for agent in agent_types]
    
    ax4.bar(agent_types, total_trades, color='orange')
    ax4.set_title('Total Trading Activity')
    ax4.set_ylabel('Number of Trades')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('agent_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def calculate_risk_metrics(performance):
    """Calculate risk-adjusted performance metrics"""
    print("\n" + "="*60)
    print("RISK-ADJUSTED PERFORMANCE METRICS")
    print("="*60)
    
    for agent_type, data in performance.items():
        if len(data['balances']) > 1:
            # Calculate returns
            returns = np.diff(data['balances'])
            
            # Risk metrics
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            
            # Sharpe ratio (assuming risk-free rate = 0)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(data['balances'])
            drawdown = (peak - data['balances']) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            print(f"\n{agent_type}:")
            print(f"  Average Return: {avg_return:.2f}")
            print(f"  Volatility: {volatility:.2f}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
            print(f"  Max Drawdown: {max_drawdown:.2%}")

def main():
    """Main analysis function"""
    # Analyze the files from our simulation
    balance_file = "bse_d001_i10_0001_avg_balance.csv"
    tape_file = "bse_d001_i10_0001_tape.csv"
    
    try:
        # Analyze performance data
        performance = analyze_avg_balance(balance_file)
        
        # Print summary
        print_performance_summary(performance)
        
        # Calculate risk metrics
        calculate_risk_metrics(performance)
        
        # Create visualizations
        plot_performance(performance)
        
        # Analyze transaction data
        transactions = analyze_tape(tape_file)
        print(f"\nTotal Transactions: {len(transactions):,}")
        
        if transactions:
            prices = [t['price'] for t in transactions]
            print(f"Price Range: ${min(prices):,} - ${max(prices):,}")
            print(f"Average Price: ${np.mean(prices):,.2f}")
            print(f"Price Volatility: ${np.std(prices):,.2f}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        print("Make sure you've run the BSE simulation first!")
        print("\nTo run a simulation:")
        print("  python BSE.py")
        print("\nTo analyze specific files:")
        print("  python analyze_performance.py [balance_file] [tape_file]")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
