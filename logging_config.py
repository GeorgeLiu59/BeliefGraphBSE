"""
BeliefGraphBSE Logging Configuration Module

This module sets up comprehensive logging for different components of the trading system.
Each component logs to its own file with detailed debug information.
"""

import logging
import os
from datetime import datetime
import json
from typing import Any, Dict


class PrintStyleFormatter(logging.Formatter):
    """
    Formatter that recreates print-style output with nice formatting.
    
    This formatter takes single-line messages and formats them to look like
    print statements in the logs, with proper spacing and structure.
    
    Example input:
        "=== DEBUG === | Line 1 | Line 2"
    
    Example output:
        14:30:25 - DEBUG - === DEBUG === | Line 1 | Line 2
    """
    
    def format(self, record):
        # Get the base message
        message = record.getMessage()
        
        # Since we're now enforcing single-line output, just use standard format
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'component': record.name,
            'message': record.getMessage(),
            'time_ms': int(record.created * 1000)
        }
        
        # Add extra fields if present
        if hasattr(record, 'trader_id'):
            log_obj['trader_id'] = record.trader_id
        if hasattr(record, 'market_time'):
            log_obj['market_time'] = record.market_time
        if hasattr(record, 'hypothesis'):
            log_obj['hypothesis'] = record.hypothesis
        if hasattr(record, 'decision'):
            log_obj['decision'] = record.decision
        if hasattr(record, 'lob_state'):
            log_obj['lob_state'] = record.lob_state
        if hasattr(record, 'order'):
            log_obj['order'] = record.order
        if hasattr(record, 'trade'):
            log_obj['trade'] = record.trade
        if hasattr(record, 'memory_state'):
            log_obj['memory_state'] = record.memory_state
        if hasattr(record, 'reasoning'):
            log_obj['reasoning'] = record.reasoning
            
        return json.dumps(log_obj)


class DetailedFormatter(logging.Formatter):
    """Human-readable detailed formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color for console output
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the base message
        result = super().format(record)
        
        # Add extra context if available
        extras = []
        if hasattr(record, 'trader_id'):
            extras.append(f"Trader: {record.trader_id}")
        if hasattr(record, 'market_time'):
            extras.append(f"Market Time: {record.market_time:.2f}")
        if hasattr(record, 'hypothesis'):
            extras.append(f"Hypothesis: {record.hypothesis}")
        if hasattr(record, 'decision'):
            extras.append(f"Decision: {record.decision}")
        
        if extras:
            result += f" | {' | '.join(extras)}"
            
        return result


def setup_logging(log_dir: str = "logs", verbose: bool = True, debug_mode: bool = True):
    """
    Set up simplified logging for BSE components - just 3 files total
    
    Args:
        log_dir: Directory for log files
        verbose: Enable verbose console output
        debug_mode: Enable debug-level logging
    """
    
    # Create log directory (no subdirectories)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for this session
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Console handler with detailed formatter
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(DetailedFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))
        root_logger.addHandler(console_handler)
    
    # Simplified 3-file logging structure
    loggers_config = {
        'BSE.LLM': {
            'file': f'{log_dir}/llm_agents.txt',
            'level': logging.DEBUG,
            'format': 'print_style'
        },
        'BSE.LLM_HM': {
            'file': f'{log_dir}/hm_agents.txt',
            'level': logging.DEBUG,
            'format': 'print_style'
        },
        'BSE': {
            'file': f'{log_dir}/general_output.txt',
            'level': logging.INFO,
            'format': 'detailed'
        }
    }
    
    for logger_name, config in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(config['level'])
        logger.propagate = False  # Don't propagate to root logger
        
        # File handler
        file_handler = logging.FileHandler(config['file'])
        file_handler.setLevel(config['level'])
        
        # Choose formatter based on config
        if config['format'] == 'json':
            formatter = JSONFormatter()
        elif config['format'] == 'print_style':
            formatter = PrintStyleFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = DetailedFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # LLM_HM logs should only go to file, not console
        # Removed console handler to prevent duplication in general_output.txt
    
    # Log initialization to console
    print(f"Logging system initialized - Session: {session_time}")
    print(f"Log directory: {os.path.abspath(log_dir)}")
    print(f"Debug mode: {debug_mode}, Verbose: {verbose}")
    
    return session_time


class TraderLogger:
    """
    Helper class for trader-specific logging with context.
    
    Now includes print_style_log() method for multi-line print-style logging!
    
    IMPORTANT: Always initialize this logger BEFORE any logging calls in trader classes.
    
    RULE: Every log message MUST be one line with pipe separators for readability.
    
    FLOW LABELING: All flow steps are now properly labeled in TraderLLM_HM class.
    """
    
    def __init__(self, trader_id: str, trader_type: str):
        self.trader_id = trader_id
        self.trader_type = trader_type
        
        # Get appropriate logger based on trader type
        if 'LLM_HM' in trader_type:
            self.logger = logging.getLogger('BSE.LLM_HM')
        elif 'LLM' in trader_type:
            self.logger = logging.getLogger('BSE.LLM')
        else:
            self.logger = logging.getLogger('BSE.Core')
    
    def log_decision(self, market_time: float, decision: Dict[str, Any], market_data: str = None):
        """Log a trading decision with full context"""
        self.print_style_log('debug', f"""=== TRADING DECISION ===
Trader: {self.trader_id}
Market Time: {market_time:.2f}
Decision: {decision}
Market Data: {market_data}""")
    
    def log_hypothesis(self, hypothesis_id: str, hypothesis_data: Dict[str, Any], 
                      evaluation_score: float = None):
        """Log hypothesis generation or evaluation"""
        # For HM traders, log hypotheses to the HM logger
        if 'LLM_HM' in self.trader_type:
            self.print_style_log('debug', f"""=== HYPOTHESIS {hypothesis_id} ===
Trader: {self.trader_id}
Data: {hypothesis_data}
Score: {evaluation_score}""")
    
    def log_order(self, order: Any, reasoning: str = None):
        """Log order creation"""
        # Log orders to the appropriate trader logger
        self.print_style_log('debug', f"""=== ORDER CREATED ===
Trader: {self.trader_id}
Order: {order}
Reasoning: {reasoning}""")
    
    def log_trade_response(self, trade: Any, response: str = None):
        """Log response to a trade"""
        # Log trade responses to the appropriate trader logger
        self.print_style_log('info', f"""=== TRADE RESPONSE ===
Trader: {self.trader_id}
Trade: {trade}
Response: {response}""")
    
    def log_memory_update(self, memory_state: Dict[str, Any]):
        """Log memory state updates for HM traders"""
        if 'LLM_HM' in self.trader_type:
            self.print_style_log('debug', f"""=== MEMORY UPDATE ===
Trader: {self.trader_id}
Memory State: {memory_state}""")
    
    def log_llm_interaction(self, prompt: str, response: str, parse_result: Any = None):
        """Log LLM API interactions"""
        self.print_style_log('debug', f"""=== LLM INTERACTION ===
Trader: {self.trader_id}
Prompt: {prompt[:200]}...
Response: {response[:200]}...
Parsed: {parse_result}""")
    
    def print_style_log(self, level: str, multi_line_message: str):
        """
        Log a multi-line message in print-style format
        
        Args:
            level: Log level ('debug', 'info', 'warning', 'error')
            multi_line_message: Multi-line string that would normally be printed
            
        Example:
            # Instead of multiple print statements, use this single call:
            # self.trader_logger.print_style_log('debug', multi_line_string)
            
        RULE: Every log message MUST be one line with pipe separators for readability
        """
        # Get the appropriate logging method
        log_method = getattr(self.logger, level.lower(), self.logger.debug)
        
        # Convert multi-line message to single line with pipe separators
        # Remove newlines and replace with pipe separators
        single_line = multi_line_message.replace('\n', ' | ').replace('  ', ' ').strip()
        
        # Log the single-line message
        log_method(single_line)
