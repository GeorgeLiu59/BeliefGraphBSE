# BSE Belief Graph Implementation Plan

## Final BSE Belief Structure (Based on Hanabi + Strategic Fields)

### Variation 1 - Pure Discrete (GraphVar1)
```json
{
  "MarketState": { 
    "current_bid": 95, 
    "current_ask": 105, 
    "time_remaining": 120 
  },
  "My_Trading_Beliefs": {
    "My_Market_Direction": {
      "actual_signals_I_see": "mixed_price_movements",
      "my_self_belief": { 
        "possible_directions": ["up", "down", "sideways"] 
      }
    },
    "My_Optimal_Entry_Price": {
      "actual_situation_I_face": "volatile_spreads",
      "my_self_belief": { 
        "possible_prices": [92, 95, 98] 
      }
    },
    "My_Time_Urgency": {
      "actual_situation_I_face": "time_pressure_building",
      "my_self_belief": {
        "possible_urgency": ["low", "medium", "high"]
      }
    }
  },
  "Competitor_Trading_Beliefs": {
    "P01_Beliefs": {
      "P01_Valuation": {
        "actual_behavior_I_observe": "bidding_at_92",
        "p01_self_belief": { 
          "possible_valuations": [85, 90, 95] 
        }
      },
      "P01_Market_Direction": {
        "actual_behavior_I_observe": "cautious_timing",
        "p01_self_belief": { 
          "possible_directions": ["down", "sideways"] 
        }
      },
      "P01_Desperation_Level": {
        "actual_behavior_I_observe": "patient_small_bids",
        "p01_self_belief": { 
          "possible_desperation": ["calm", "moderate", "desperate"] 
        }
      },
      "P01_Available_Cash": {
        "actual_behavior_I_observe": "consistent_small_volumes",
        "p01_self_belief": {
          "possible_cash": ["low", "medium", "high"]
        }
      },
      "P01_Exit_Strategy": {
        "actual_behavior_I_observe": "holding_positions",
        "p01_self_belief": {
          "possible_exits": ["hold_till_end", "sell_early", "opportunistic"]
        }
      }
    },
    "P02_Beliefs": {
      "P02_Valuation": {
        "actual_behavior_I_observe": "aggressive_asking_at_108",
        "p02_self_belief": {
          "possible_valuations": [75, 80, 85]
        }
      },
      "P02_Market_Direction": {
        "actual_behavior_I_observe": "urgent_selling",
        "p02_self_belief": {
          "possible_directions": ["up", "sideways"]
        }
      },
      "P02_Desperation_Level": {
        "actual_behavior_I_observe": "large_urgent_orders",
        "p02_self_belief": {
          "possible_desperation": ["moderate", "desperate"]
        }
      }
    }
  }
}
```

### Variation 2 - Pure Probabilistic (GraphVar2)
```json
{
  "MarketState": { 
    "current_bid": 95, 
    "current_ask": 105, 
    "time_remaining": 120 
  },
  "My_Trading_Beliefs": {
    "My_Market_Direction": {
      "actual_signals_I_see": "mixed_price_movements",
      "my_self_belief": {
        "direction_distribution": {"up": 0.4, "down": 0.3, "sideways": 0.3}
      }
    },
    "My_Optimal_Entry_Price": {
      "actual_situation_I_face": "volatile_spreads", 
      "my_self_belief": {
        "price_distribution": {"92": 0.3, "95": 0.4, "98": 0.3}
      }
    },
    "My_Time_Urgency": {
      "actual_situation_I_face": "time_pressure_building",
      "my_self_belief": {
        "urgency_distribution": {"low": 0.5, "medium": 0.3, "high": 0.2}
      }
    }
  },
  "Competitor_Trading_Beliefs": {
    "P01_Beliefs": {
      "P01_Valuation": {
        "actual_behavior_I_observe": "bidding_at_92",
        "p01_self_belief": {
          "valuation_distribution": {"85": 0.4, "90": 0.3, "95": 0.3}
        }
      },
      "P01_Market_Direction": {
        "actual_behavior_I_observe": "cautious_timing",
        "p01_self_belief": {
          "direction_distribution": {"down": 0.6, "sideways": 0.4, "up": 0.0}
        }
      },
      "P01_Desperation_Level": {
        "actual_behavior_I_observe": "patient_small_bids",
        "p01_self_belief": {
          "desperation_distribution": {"calm": 0.7, "moderate": 0.3, "desperate": 0.0}
        }
      },
      "P01_Available_Cash": {
        "actual_behavior_I_observe": "consistent_small_volumes",
        "p01_self_belief": {
          "cash_distribution": {"low": 0.6, "medium": 0.3, "high": 0.1}
        }
      }
    }
  }
}
```

## Implementation Tasks

### 1. Modify GraphVar1 Methods (Pure Discrete)
- **`_add_initial_beliefs(agent_id)`**: Create discrete possible value sets for each belief type
- **`_update_beliefs_from_bid/ask/trade()`**: Use set elimination logic (like Hanabi clue elimination)
- **`query_action()`**: Return discrete possibilities in new JSON structure
- **Remove confidence mixing**: No more single values + confidence scores

### 2. Modify GraphVar2 Methods (Pure Probabilistic)  
- **`_add_initial_beliefs(agent_id)`**: Create uniform probability distributions
- **`_update_beliefs_from_bid/ask/trade()`**: Bayesian probability updates based on market events
- **`query_action()`**: Return probability distributions in new JSON structure
- **Pure distributions**: No confidence scores, only probability weights

### 3. Strategic Field Focus
High-impact fields for BSE success:
- **Valuation beliefs** (competitor valuation uncertainty → arbitrage opportunities)
- **Market direction beliefs** (prediction disagreements → timing advantages)  
- **Desperation/urgency levels** (timing pressure modeling → panic trade exploitation)
- **Entry price optimization** (tactical bidding decisions → profit maximization)
- **Available cash** (resource constraints → competitive positioning)
- **Exit strategy** (hold vs sell timing → market impact prediction)

### 4. BSE.py Integration Updates
- Detect trader type and route to correct GraphVar1/GraphVar2 class
- Minimal changes - leverage existing `query_action()` architecture
- Update imports if needed

### 5. Update Logic (Hanabi-Style)
- **Discrete elimination**: Remove impossible values based on market events
- **Probabilistic reweighting**: Bayesian updates on distributions
- **Recursive Theory of Mind**: Track "what I think P01 thinks about themselves"
- **Market event inference**: Bid/ask/trade events eliminate possibilities or update probabilities

## Key Principles
- **Pure to specification**: 1:1 correspondence with Rahul's Hanabi structure
- **No confidence mixing**: Discrete sets OR probability distributions, not both
- **Strategic focus**: Fields that create real arbitrage opportunities in BSE
- **Symmetric structure**: Consistent naming between My_Beliefs and Competitor_Beliefs
- **Edit existing methods**: No new files, modify current GraphVar1/GraphVar2 implementations
- **Recursive beliefs**: "My beliefs about myself" vs "My beliefs about what P01 thinks about themselves"

## Success Metrics
- LLM agents can exploit valuation gaps between competitors
- Market direction disagreements create timing advantages
- Desperation level tracking enables panic trade exploitation  
- Entry price optimization improves profit margins
- Resource constraint awareness improves competitive positioning