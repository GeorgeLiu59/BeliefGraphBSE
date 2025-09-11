#!/usr/bin/env python3
"""
Trader with LLM-Designed Custom Attributes

This trader uses the LLM-designed custom attributes system instead of
predefined attributes. The LLM invents its own trading personality dimensions.
"""

import os
import sys
import time
import random
from dotenv import load_dotenv
from llm_designed_attributes import CustomAttributeManager, CustomAttributeSet

# Add current directory to path
sys.path.append('.')

class TraderCustomAttributes:
    """Trader that uses LLM-designed custom attributes"""
    
    def __init__(self, ttype: str, tid: str, balance: float, params, time, use_attributes: bool = True, single_attribute: str = None):
        """Initialize the custom attributes trader"""
        # Initialize basic trader properties (compatible with BSE Trader)
        self.ttype = ttype
        self.tid = tid
        self.balance = balance
        self.params = params
        self.birthtime = time
        self.verbose = True  # Enable verbose output for custom attributes
        self.use_attributes = use_attributes  # Control whether to use attributes or not
        self.single_attribute = single_attribute  # If set, use only this predefined attribute
        
        # Load environment variables
        load_dotenv()
        
        # Initialize LLM model (like HM LLM does)
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.model = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
                if self.verbose:
                    print(f"Custom Attributes Trader {tid}: LLM model initialized successfully")
            except Exception as e:
                if self.verbose:
                    print(f"Custom Attributes Trader {tid}: Failed to initialize LLM model: {e}")
        else:
            if self.verbose:
                print(f"Custom Attributes Trader {tid}: No API key found, LLM disabled")
        
        # Initialize custom attribute manager
        self.custom_attribute_manager = CustomAttributeManager(f"{tid}_{ttype}")
        self.attributes_initialized = False
        self.design_rationale = ""  # Initialize design rationale
        
        # Trading state tracking (like HM LLM)
        self.inventory = 0  # how many units we currently hold
        self.last_purchase_price = None
        self.total_profit = 0.0
        self.trade_count = 0
        
        # BSE simulation compatibility
        self.orders = []
        self.profitpertime = 0.0
        self.blotter = []  # Required by BSE simulation
        
        # Market context for attribute design
        self.market_context = {
            'volatility': 'Unknown',
            'trend': 'Unknown',
            'competition': 'Unknown',
            'liquidity': 'Unknown'
        }
        
        if self.verbose:
            print(f"Initialized Custom Attributes Trader {tid} with balance ${balance}")
    
    def profitpertime_update(self, time, birthtime, totalprofit):
        """Update profit per time metric (required by BSE simulation)"""
        if time > birthtime:
            return totalprofit / (time - birthtime)
        return 0.0
    
    def initialize_custom_attributes(self, market_context: dict = None):
        """Initialize custom attributes using LLM"""
        if not self.use_attributes:
            # No attributes mode - just mark as initialized
            self.attributes_initialized = True
            self.design_rationale = "No attributes mode - pure LLM decisions"
            if self.verbose:
                print(f"Custom Attributes Trader {self.tid}: No attributes mode enabled")
            return
        
        if self.single_attribute:
            # Single attribute mode - use predefined attribute
            self.attributes_initialized = True
            self.design_rationale = f"Single attribute mode - using {self.single_attribute}"
            if self.verbose:
                print(f"Custom Attributes Trader {self.tid}: Single attribute mode enabled - {self.single_attribute}")
            return
        
        if market_context:
            self.market_context.update(market_context)
        
        try:
            self.custom_attribute_manager.initialize_custom_attributes(self.market_context)
            self.attributes_initialized = True
            
            # Get the attributes and set the design rationale
            attributes = self.custom_attribute_manager.get_attributes()
            self.design_rationale = attributes.design_rationale
            
            if self.verbose:
                print(f"Custom Attributes Trader {self.tid} designed attributes:")
                print(f"Rationale: {self.design_rationale}")
                for attr in attributes.attributes:
                    print(f"  • {attr.name}: {attr.value:.2f} - {attr.description}")
            
        except Exception as e:
            print(f"Error initializing custom attributes: {e}")
            self.attributes_initialized = False
    
    def get_custom_attributes(self) -> CustomAttributeSet:
        """Get current custom attributes"""
        if not self.attributes_initialized:
            raise ValueError("Custom attributes not initialized. Call initialize_custom_attributes() first.")
        return self.custom_attribute_manager.get_attributes()
    
    def adapt_custom_attributes(self, performance_metrics: dict):
        """Adapt custom attributes based on performance"""
        if not self.attributes_initialized:
            return
        
        # Update market context with performance metrics
        context = self.market_context.copy()
        context.update(performance_metrics)
        
        try:
            new_attributes = self.custom_attribute_manager.adapt_custom_attributes(context)
            if new_attributes and self.verbose:
                print(f"Custom Attributes Trader {self.tid} adapted attributes:")
                print(f"New Rationale: {new_attributes.design_rationale}")
        except Exception as e:
            if self.verbose:
                print(f"Error adapting custom attributes: {e}")
    
    def get_trading_decision(self, current_price: float, market_data: dict = None) -> tuple[str, float]:
        """Make trading decision based on custom attributes"""
        if not self.attributes_initialized:
            return "WAIT", 0.0
        
        try:
            attributes = self.get_custom_attributes()
            
            # Get relevant attribute values
            trend_following = attributes.get_attribute_value("trend_following")
            momentum_aggressiveness = attributes.get_attribute_value("momentum_aggressiveness")
            patience_level = attributes.get_attribute_value("patience_level")
            volatility_tolerance = attributes.get_attribute_value("volatility_tolerance")
            range_recognition = attributes.get_attribute_value("range_recognition")
            competition_awareness = attributes.get_attribute_value("competition_awareness")
            
            # Default values if attributes don't exist
            if trend_following == 0.5:  # Default fallback value
                trend_following = attributes.get_attribute_value("trend_sensitivity")
            if momentum_aggressiveness == 0.5:
                momentum_aggressiveness = attributes.get_attribute_value("aggressiveness")
            
            # Simple trading logic based on custom attributes
            decision = self._make_decision_with_custom_attributes(
                current_price, 
                trend_following,
                momentum_aggressiveness,
                patience_level,
                volatility_tolerance,
                range_recognition,
                competition_awareness
            )
            
            return decision
            
        except Exception as e:
            if self.verbose:
                print(f"Error in trading decision: {e}")
            return "WAIT", 0.0
    
    def _make_decision_with_custom_attributes(
        self, 
        current_price: float,
        trend_following: float,
        momentum_aggressiveness: float,
        patience_level: float,
        volatility_tolerance: float,
        range_recognition: float,
        competition_awareness: float
    ) -> tuple[str, float]:
        """Make trading decision using custom attributes AND LLM (like HM LLM)"""
        
        # Create market context for LLM (similar to HM LLM)
        market_context = {
            'current_price': current_price,
            'inventory': self.inventory,
            'balance': self.balance,
            'total_profit': self.total_profit,
            'trade_count': self.trade_count,
            'custom_attributes': {
                'trend_following': trend_following,
                'momentum_aggressiveness': momentum_aggressiveness,
                'patience_level': patience_level,
                'volatility_tolerance': volatility_tolerance,
                'range_recognition': range_recognition,
                'competition_awareness': competition_awareness
            },
            'design_rationale': self.design_rationale
        }
        
        # Get LLM trading decision (like HM LLM does)
        try:
            llm_decision = self._get_llm_trading_decision(market_context)
            if llm_decision:
                return llm_decision
        except Exception as e:
            if self.verbose:
                print(f"LLM decision failed: {e}, using fallback logic")
        
        # Fallback to mathematical logic if LLM fails
        return self._fallback_decision_with_attributes(
            current_price, trend_following, momentum_aggressiveness, 
            patience_level, volatility_tolerance, range_recognition, competition_awareness
        )
    
    def _get_llm_trading_decision(self, market_context: dict) -> dict:
        """Get trading decision from LLM based on custom attributes (similar to HM LLM)"""
        if not hasattr(self, 'model') or not self.model:
            return self._fallback_decision()
        
        # Format market context similar to HM LLM
        current_price = market_context.get('current_price', 0)
        best_bid = market_context.get('best_bid', 'None')
        best_ask = market_context.get('best_ask', 'None')
        
        # Create different prompts based on whether we use attributes or not
        if self.single_attribute:
            # Single attribute mode - use predefined attribute
            if self.inventory == 0:  # Looking to BUY
                prompt = f"""You are a proprietary trader with ${self.balance} trying to make profit by buying low and selling high.

CURRENT MARKET SITUATION:
- Current Price: ${current_price}
- Best Bid: {best_bid}
- Best Ask: {best_ask}
- Market Volatility: {self.market_context.get('volatility', 'Unknown')}
- Market Trend: {self.market_context.get('trend', 'Unknown')}
- Competition Level: {self.market_context.get('competition', 'Unknown')}
- Liquidity: {self.market_context.get('liquidity', 'Unknown')}

MY TRADING PERSONALITY:
- {self.single_attribute}: 0.7 (moderate level) - This is my main trading characteristic

CURRENT SITUATION: You currently have NO INVENTORY and are looking to BUY a unit.

IMPORTANT: You MUST make a profit. Your goal is to end with MORE money than you started with.

TRADING PRINCIPLES BASED ON MY {self.single_attribute.upper()}:
- My {self.single_attribute} level (0.7) means I am moderately aggressive in my trading approach
- With moderate aggressiveness, I should bid SIGNIFICANTLY HIGHER than market price to ensure execution
- I should be willing to pay 5-10% above market price to secure trades immediately
- I should act decisively and bid aggressively rather than being overly cautious
- EXAMPLE: If market price is $150, I should bid $160-165 to ensure my order executes

HOW ORDER BOOKS WORK:
- To BUY: Place a BID order at your desired price
- If sellers exist at/below your bid price → immediate execution
- If no sellers at your price → your bid waits on the order book for sellers
- Higher bids are more likely to execute quickly

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""

            else:  # Looking to SELL
                prompt = f"""You are a proprietary trader trying to make profit by buying low and selling high.

CURRENT MARKET SITUATION:
- Current Price: ${current_price}
- Best Bid: {best_bid}
- Best Ask: {best_ask}
- Market Volatility: {self.market_context.get('volatility', 'Unknown')}
- Market Trend: {self.market_context.get('trend', 'Unknown')}
- Competition Level: {self.market_context.get('competition', 'Unknown')}
- Liquidity: {self.market_context.get('liquidity', 'Unknown')}

MY TRADING PERSONALITY:
- {self.single_attribute}: 0.7 (moderate level) - This is my main trading characteristic

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

IMPORTANT: You MUST make a profit. Your goal is to end with MORE money than you started with ($500).

TRADING PRINCIPLES BASED ON MY {self.single_attribute.upper()}:
- My {self.single_attribute} level (0.7) means I am moderately aggressive in my trading approach
- With moderate aggressiveness, I should ask SIGNIFICANTLY LOWER than market price to ensure execution
- I should be willing to accept 5-10% below market price to secure trades immediately
- I should act decisively and ask aggressively rather than being overly cautious
- EXAMPLE: If market price is $150, I should ask $135-140 to ensure my order executes

HOW ORDER BOOKS WORK:
- To SELL: Place an ASK order at your desired price
- If buyers exist at/above your ask price → immediate execution  
- If no buyers at your price → your ask waits on the order book for buyers
- Lower asks are more likely to execute quickly

PROFIT/LOSS ANALYSIS:
- You bought at: ${self.last_purchase_price}
- Break-even price: ${self.last_purchase_price}
- To profit: sell above ${self.last_purchase_price}
- Current best bid: {best_bid} (immediate execution if you ask at/below this)

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price  
"WAIT" - to wait for better conditions

No explanation needed."""

        elif not self.use_attributes:
            # No attributes mode - pure LLM decision making
            if self.inventory == 0:  # Looking to BUY
                prompt = f"""You are a proprietary trader with ${self.balance} trying to make profit by buying low and selling high.

CURRENT MARKET SITUATION:
- Current Price: ${current_price}
- Best Bid: {best_bid}
- Best Ask: {best_ask}
- Market Volatility: {self.market_context.get('volatility', 'Unknown')}
- Market Trend: {self.market_context.get('trend', 'Unknown')}
- Competition Level: {self.market_context.get('competition', 'Unknown')}
- Liquidity: {self.market_context.get('liquidity', 'Unknown')}

CURRENT SITUATION: You currently have NO INVENTORY and are looking to BUY a unit.

IMPORTANT: You MUST make a profit. Your goal is to end with MORE money than you started with.

HOW ORDER BOOKS WORK:
- To BUY: Place a BID order at your desired price
- If sellers exist at/below your bid price → immediate execution
- If no sellers at your price → your bid waits on the order book for sellers
- Higher bids are more likely to execute quickly

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""

            else:  # Looking to SELL
                prompt = f"""You are a proprietary trader trying to make profit by buying low and selling high.

CURRENT MARKET SITUATION:
- Current Price: ${current_price}
- Best Bid: {best_bid}
- Best Ask: {best_ask}
- Market Volatility: {self.market_context.get('volatility', 'Unknown')}
- Market Trend: {self.market_context.get('trend', 'Unknown')}
- Competition Level: {self.market_context.get('competition', 'Unknown')}
- Liquidity: {self.market_context.get('liquidity', 'Unknown')}

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

IMPORTANT: You MUST make a profit. Your goal is to end with MORE money than you started with ($500).

HOW ORDER BOOKS WORK:
- To SELL: Place an ASK order at your desired price
- If buyers exist at/above your ask price → immediate execution  
- If no buyers at your price → your ask waits on the order book for buyers
- Lower asks are more likely to execute quickly

PROFIT/LOSS ANALYSIS:
- You bought at: ${self.last_purchase_price}
- Break-even price: ${self.last_purchase_price}
- To profit: sell above ${self.last_purchase_price}
- Current best bid: {best_bid} (immediate execution if you ask at/below this)

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price  
"WAIT" - to wait for better conditions

No explanation needed."""

        else:
            # Attributes mode - use custom attributes
            # Get current custom attributes
            attributes = self.custom_attribute_manager.get_attributes()
            
            # Create detailed prompt incorporating custom attributes
            if self.inventory == 0:  # Looking to BUY
                prompt = f"""You are a proprietary trader with ${self.balance} trying to make profit by buying low and selling high.

CURRENT MARKET SITUATION:
- Current Price: ${current_price}
- Best Bid: {best_bid}
- Best Ask: {best_ask}
- Market Volatility: {self.market_context.get('volatility', 'Unknown')}
- Market Trend: {self.market_context.get('trend', 'Unknown')}
- Competition Level: {self.market_context.get('competition', 'Unknown')}
- Liquidity: {self.market_context.get('liquidity', 'Unknown')}

MY CUSTOM TRADING PERSONALITY (designed by me):
- trend_following: {attributes.attributes[0].value:.2f} - {attributes.attributes[0].description}
- volatility_awareness: {attributes.attributes[1].value:.2f} - {attributes.attributes[1].description}
- aggressive_entry: {attributes.attributes[2].value:.2f} - {attributes.attributes[2].description}
- profit_target_reach: {attributes.attributes[3].value:.2f} - {attributes.attributes[3].description}

MY TRADING RATIONALE: {attributes.design_rationale}

CURRENT SITUATION: You currently have NO INVENTORY and are looking to BUY a unit.

IMPORTANT: You MUST make a profit. Your goal is to end with MORE money than you started with.

TRADING PRINCIPLES BASED ON MY ATTRIBUTES:
- My trend_following ({attributes.attributes[0].value:.2f}) affects how I respond to market trends
- My volatility_awareness ({attributes.attributes[1].value:.2f}) influences my risk tolerance
- My aggressive_entry ({attributes.attributes[2].value:.2f}) determines how quickly I enter trades
- My profit_target_reach ({attributes.attributes[3].value:.2f}) affects when I take profits

HOW ORDER BOOKS WORK:
- To BUY: Place a BID order at your desired price
- If sellers exist at/below your bid price → immediate execution
- If no sellers at your price → your bid waits on the order book for sellers
- Higher bids are more likely to execute quickly

Respond with ONLY:
"BUY [exact_price]" - to place a bid at that price
"WAIT" - to wait for better conditions

No explanation needed."""

            else:  # Looking to SELL
                prompt = f"""You are a proprietary trader trying to make profit by buying low and selling high.

CURRENT MARKET SITUATION:
- Current Price: ${current_price}
- Best Bid: {best_bid}
- Best Ask: {best_ask}
- Market Volatility: {self.market_context.get('volatility', 'Unknown')}
- Market Trend: {self.market_context.get('trend', 'Unknown')}
- Competition Level: {self.market_context.get('competition', 'Unknown')}
- Liquidity: {self.market_context.get('liquidity', 'Unknown')}

MY CUSTOM TRADING PERSONALITY (designed by me):
- trend_following: {attributes.attributes[0].value:.2f} - {attributes.attributes[0].description}
- volatility_awareness: {attributes.attributes[1].value:.2f} - {attributes.attributes[1].description}
- aggressive_entry: {attributes.attributes[2].value:.2f} - {attributes.attributes[2].description}
- profit_target_reach: {attributes.attributes[3].value:.2f} - {attributes.attributes[3].description}

MY TRADING RATIONALE: {attributes.design_rationale}

CURRENT SITUATION: You are holding 1 unit that you bought for ${self.last_purchase_price}. You need to SELL it.

IMPORTANT: You MUST make a profit. Your goal is to end with MORE money than you started with ($500).

TRADING PRINCIPLES BASED ON MY ATTRIBUTES:
- My trend_following ({attributes.attributes[0].value:.2f}) affects how I respond to market trends
- My volatility_awareness ({attributes.attributes[1].value:.2f}) influences my risk tolerance
- My aggressive_entry ({attributes.attributes[2].value:.2f}) determines how quickly I enter trades
- My profit_target_reach ({attributes.attributes[3].value:.2f}) affects when I take profits

HOW ORDER BOOKS WORK:
- To SELL: Place an ASK order at your desired price
- If buyers exist at/above your ask price → immediate execution  
- If no buyers at your price → your ask waits on the order book for buyers
- Lower asks are more likely to execute quickly

PROFIT/LOSS ANALYSIS:
- You bought at: ${self.last_purchase_price}
- Break-even price: ${self.last_purchase_price}
- To profit: sell above ${self.last_purchase_price}
- Current best bid: {best_bid} (immediate execution if you ask at/below this)

Respond with ONLY:
"SELL [exact_price]" - to place an ask at that price  
"WAIT" - to wait for better conditions

No explanation needed."""

        try:
            import google.generativeai as genai
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=200
                )
            )
            
            response_text = response.text.strip()
            return self._parse_llm_response(response_text)
            
        except Exception as e:
            print(f"LLM response parsing failed: {e}")
            return self._fallback_decision()
    
    def _parse_llm_response(self, response_text):
        """Parse LLM response into actionable decision (similar to HM LLM)"""
        response_upper = response_text.upper()
        
        # Look for explicit decision patterns first (more specific)
        import re
        
        # Check for explicit BUY command with price
        buy_match = re.search(r'BUY\s+(\d+(?:\.\d+)?)', response_upper)
        if buy_match and self.inventory == 0:
            price = float(buy_match.group(1))
            # Ensure price is within valid bounds (allow higher prices for BTC market)
            price = max(1, min(200000, price))
            return {
                'action': 'BUY',
                'price': price,
                'reasoning': response_text
            }
        
        # Check for explicit SELL command with price
        sell_match = re.search(r'SELL\s+(\d+(?:\.\d+)?)', response_upper)
        if sell_match and self.inventory > 0:
            price = float(sell_match.group(1))
            # Ensure price is within valid bounds (allow higher prices for BTC market)
            price = max(1, min(200000, price))
            return {
                'action': 'SELL',
                'price': price,
                'reasoning': response_text
            }
        
        # If no explicit price commands found, default to WAIT
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': response_text
        }
    
    def _fallback_decision(self):
        """Simple fallback decision if LLM is unavailable"""
        return {
            'action': 'WAIT',
            'price': None,
            'reasoning': 'LLM unavailable, using fallback wait strategy'
        }
    
    def _fallback_decision_with_attributes(
        self, 
        current_price: float,
        trend_following: float,
        momentum_aggressiveness: float,
        patience_level: float,
        volatility_tolerance: float,
        range_recognition: float,
        competition_awareness: float
    ) -> tuple[str, float]:
        """Fallback decision logic using custom attributes"""
        
        # Determine if we should trade based on patience level
        if patience_level > 0.7 and random.random() < 0.3:
            return "WAIT", 0.0  # High patience = wait more often
        
        # Determine aggressiveness
        base_aggressiveness = (trend_following + momentum_aggressiveness) / 2
        
        # Adjust for volatility tolerance
        if volatility_tolerance < 0.3:
            base_aggressiveness *= 0.5  # Low volatility tolerance = less aggressive
        
        # Determine action based on inventory and attributes
        if self.inventory == 0:  # No inventory, consider buying
            if base_aggressiveness > 0.6:
                # High aggressiveness = buy at current price or slightly above
                buy_price = current_price * (1 + random.uniform(0, 0.02))
                return "BUY", buy_price
            elif base_aggressiveness > 0.3:
                # Medium aggressiveness = buy at slight discount
                buy_price = current_price * (1 - random.uniform(0, 0.01))
                return "BUY", buy_price
            else:
                # Low aggressiveness = wait for better price
                return "WAIT", 0.0
        
        else:  # Have inventory, consider selling
            if base_aggressiveness > 0.6:
                # High aggressiveness = sell at current price or slightly below
                sell_price = current_price * (1 - random.uniform(0, 0.02))
                return "SELL", sell_price
            elif base_aggressiveness > 0.3:
                # Medium aggressiveness = sell at slight premium
                sell_price = current_price * (1 + random.uniform(0, 0.01))
                return "SELL", sell_price
            else:
                # Low aggressiveness = wait for better price
                return "WAIT", 0.0
    
    def respond(self, time, lob, trade, vrbs):
        """
        Respond to market events and make trading decisions using LLM + custom attributes
        This method is called by the BSE simulation (similar to HM LLM)
        """
        if not self.attributes_initialized:
            return None

        # Update profit per time metric
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.total_profit)

        # Add throttling to reduce API calls (avoid quota exhaustion)
        if not hasattr(self, 'last_llm_call_time'):
            self.last_llm_call_time = 0
            self.llm_call_interval = 20.0  # Call LLM every 20 seconds of simulation time
        
        # Only call LLM if enough time has passed
        if (time - self.last_llm_call_time) < self.llm_call_interval:
            return None
        
        self.last_llm_call_time = time

        # Get current market price from the orderbook
        current_price = 0.0
        best_bid = None
        best_ask = None
        
        if lob['bids']['n'] > 0 and lob['asks']['n'] > 0:
            current_price = (lob['bids']['best'] + lob['asks']['best']) / 2
            best_bid = lob['bids']['best']
            best_ask = lob['asks']['best']
        elif lob['bids']['n'] > 0:
            current_price = lob['bids']['best']
            best_bid = lob['bids']['best']
        elif lob['asks']['n'] > 0:
            current_price = lob['asks']['best']
            best_ask = lob['asks']['best']
        else:
            return None
        
        # Format market context for LLM
        market_context = {
            'current_price': current_price,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'inventory': self.inventory,
            'balance': self.balance,
            'total_profit': self.total_profit,
            'trade_count': self.trade_count
        }
        
        # Get LLM decision based on custom attributes
        decision = self._get_llm_trading_decision(market_context)
        
        if self.verbose:
            print(f"Custom Attributes Trader {self.tid}: Market price ${current_price:.2f}, LLM decided {decision['action']} at time {time:.1f}")
        
        # Execute the decision
        if decision['action'] == 'BUY' and self.inventory == 0:
            self._execute_buy_decision(decision, lob, time)
        elif decision['action'] == 'SELL' and self.inventory > 0:
            self._execute_sell_decision(decision, lob, time)
        
        return None  # Orders are handled internally

    def _execute_buy_decision(self, decision, lob, time):
        """Execute a buy decision with LLM-specified price"""
        if lob['asks']['n'] == 0:
            return  # No asks available

        # Ensure we have a valid price from LLM
        buy_price = decision['price']
        if buy_price is None:
            if self.verbose:
                print(f"{self.tid} BUY decision has no price, skipping")
            return
        
        # Only safety check: can we afford it?
        if buy_price <= self.balance:
            # Import Order class locally to avoid circular import
            import BSE
            order = BSE.Order(self.tid, 'Bid', buy_price, 1, time, lob['QID'])
            self.orders = [order]
            
            if self.verbose:
                print(f"Custom Attributes Trader {self.tid} PLACED BUY ORDER at ${buy_price:.2f} | Balance: ${self.balance:.2f}")
        elif self.verbose:
            print(f"{self.tid} cannot afford price {buy_price}, balance is {self.balance}")

    def _execute_sell_decision(self, decision, lob, time):
        """Execute a sell decision with LLM-specified price"""
        if lob['bids']['n'] == 0:
            return  # No bids available

        # Ensure we have a valid price from LLM
        sell_price = decision['price']
        if sell_price is None:
            if self.verbose:
                print(f"{self.tid} SELL decision has no price, skipping")
            return
        
        # Create the order at LLM's chosen price
        import BSE
        order = BSE.Order(self.tid, 'Ask', sell_price, 1, time, lob['QID'])
        self.orders = [order]
        
        if self.verbose:
            profit = sell_price - self.last_purchase_price if self.last_purchase_price else 0
            print(f"Custom Attributes Trader {self.tid} PLACED SELL ORDER at ${sell_price:.2f} | Expected Profit: ${profit:.2f} | Balance: ${self.balance:.2f}")

    def getorder(self, time, time_left, lob):
        """
        BSE simulation expects this method name
        """
        return self.respond(time, lob, None, False)

    def get_order(self, current_time: int, current_price: float, market_data: dict = None) -> dict:
        """Get trading order based on custom attributes"""
        if not self.attributes_initialized:
            return None
        
        # Get trading decision
        action, price = self.get_trading_decision(current_price, market_data)
        
        if action == "WAIT":
            return None
        
        # Create order based on decision
        if action == "BUY" and self.balance >= price:
            order = {
                'type': 'BID',
                'price': price,
                'quantity': 1,
                'trader_id': self.tid,
                'timestamp': current_time
            }
            
            if self.verbose:
                print(f"Custom Attributes Trader {self.tid} BOUGHT at ${price:.2f} | Balance: ${self.balance:.2f}")
            
            return order
        
        elif action == "SELL" and self.inventory > 0:
            order = {
                'type': 'ASK',
                'price': price,
                'quantity': 1,
                'trader_id': self.tid,
                'timestamp': current_time
            }
            
            if self.verbose:
                profit = price - self.purchase_price if self.purchase_price > 0 else 0
                emoji = "🟢" if profit >= 0 else "🔴"
                print(f"{emoji} Custom Attributes Trader {self.tid} SOLD at ${price:.2f} | Profit: ${profit:.2f} | Total Profit: ${self.total_profit:.2f}")
            
            return order
        
        return None
    
    def process_trade(self, trade_price: float, trade_quantity: int, trade_type: str):
        """Process a completed trade"""
        if trade_type == "BUY":
            self.balance -= trade_price * trade_quantity
            self.inventory += trade_quantity
            self.purchase_price = trade_price
            self.trade_count += 1
            
        elif trade_type == "SELL":
            self.balance += trade_price * trade_quantity
            self.inventory -= trade_quantity
            if self.purchase_price > 0:
                profit = (trade_price - self.purchase_price) * trade_quantity
                self.total_profit += profit
            self.trade_count += 1
    
    def get_net_worth(self) -> float:
        """Get current net worth"""
        if self.inventory > 0 and self.purchase_price > 0:
            return self.balance + (self.inventory * self.purchase_price)
        return self.balance
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics for attribute adaptation"""
        return {
            'profit': self.total_profit,
            'trade_count': self.trade_count,
            'balance': self.balance,
            'inventory': self.inventory,
            'market_volatility': random.uniform(0.1, 0.9),  # Placeholder
            'relative_performance': random.uniform(-0.5, 0.5)  # Placeholder
        }
    
    def to_json(self) -> str:
        """Convert trader to JSON"""
        return self.custom_attribute_manager.to_json()
    
    @classmethod
    def from_json(cls, json_str: str, tid: str, name: str, balance: float, orderbook, verbose: bool = False):
        """Create trader from JSON"""
        trader = cls(tid, name, balance, orderbook, verbose)
        trader.custom_attribute_manager = CustomAttributeManager.from_json(json_str)
        trader.attributes_initialized = True
        return trader

# Example usage and testing
def test_custom_attributes_trader():
    """Test the custom attributes trader"""
    print("Testing Custom Attributes Trader")
    print("=" * 50)
    
    # Create trader
    trader = TraderCustomAttributes("CUSTOM_001", "TestTrader", 1000.0, None, verbose=True)
    
    # Initialize with different market conditions
    market_conditions = [
        {
            'volatility': 'Low',
            'trend': 'Sideways',
            'competition': 'Low',
            'liquidity': 'High'
        },
        {
            'volatility': 'High',
            'trend': 'Upward',
            'competition': 'Moderate',
            'liquidity': 'High'
        }
    ]
    
    for i, context in enumerate(market_conditions, 1):
        print(f"\nTesting Market Condition {i}: {context['volatility']} volatility, {context['trend']} trend")
        print("-" * 60)
        
        # Initialize attributes
        trader.initialize_custom_attributes(context)
        
        # Test trading decisions
        current_price = 150.0
        for step in range(5):
            order = trader.get_order(step, current_price)
            if order:
                print(f"Step {step}: {order['type']} at ${order['price']:.2f}")
                # Simulate trade execution
                trader.process_trade(order['price'], 1, order['type'])
            else:
                print(f"Step {step}: WAIT")
            
            current_price += random.uniform(-2, 2)  # Simulate price movement
        
        print(f"Final Balance: ${trader.balance:.2f}, Inventory: {trader.inventory}, Profit: ${trader.total_profit:.2f}")

    def bookkeep(self, time, trade, order, vrbs):
        """
        Update trader state when a trade is executed (required by BSE simulation)
        This method is called by the BSE simulation when our order gets filled
        """
        if not trade or not order:
            return
        
        transactionprice = trade['price']
        
        if self.verbose:
            print(f"Custom Attributes Trader {self.tid}: Trade executed at ${transactionprice:.2f}")
        
        if order.otype == 'Bid':
            # Successfully bought a unit
            self.balance -= transactionprice
            self.last_purchase_price = transactionprice
            self.inventory = 1
            if self.verbose:
                print(f"Custom Attributes Trader {self.tid}: BOUGHT at ${transactionprice:.2f} | New Balance: ${self.balance:.2f}")
                
        elif order.otype == 'Ask':
            # Successfully sold a unit
            old_balance = self.balance
            self.balance += transactionprice
            if self.last_purchase_price is not None:
                profit = transactionprice - self.last_purchase_price
                self.total_profit += profit
                if self.verbose:
                    print(f"Custom Attributes Trader {self.tid}: SOLD at ${transactionprice:.2f} | Profit: ${profit:.2f} | New Balance: ${self.balance:.2f}")
            else:
                if self.verbose:
                    print(f"Custom Attributes Trader {self.tid}: SOLD at ${transactionprice:.2f} | New Balance: ${self.balance:.2f}")
            
            self.inventory = 0
            self.last_purchase_price = None
        
        # Update trade count and profit per time
        self.trade_count += 1
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.total_profit)
        
        # Clear the executed order
        self.orders = []

if __name__ == "__main__":
    test_custom_attributes_trader()
