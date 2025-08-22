"""
BeliefGraphBSE Logging Configuration Module

Simplified logging setup for trading system components.
Only includes actually used functions - dead code removed.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict


class PrintStyleFormatter(logging.Formatter):
    """
    Formatter that recreates print-style output with nice formatting.
    """
    
    def format(self, record):
        # Get the base message and use standard format
        return super().format(record)


def setup_logging(log_dir: str = "logs", verbose: bool = True, debug_mode: bool = True):
    """
    Set up simplified logging for BSE components
    
    Args:
        log_dir: Directory for log files
        verbose: Enable verbose console output
        debug_mode: Enable debug-level logging
    """
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for this session
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Console handler if verbose
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))
        root_logger.addHandler(console_handler)
    
    # File logging configuration
    loggers_config = {
        'BSE.LLM': {
            'file': f'{log_dir}/llm_agents.txt',
            'level': logging.DEBUG,
        },
        'BSE.LLM_HM': {
            'file': f'{log_dir}/hm_agents.txt',
            'level': logging.DEBUG,
        },
        'BSE': {
            'file': f'{log_dir}/general_output.txt',
            'level': logging.INFO,
        }
    }
    
    for logger_name, config in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(config['level'])
        logger.propagate = False
        
        # File handler
        file_handler = logging.FileHandler(config['file'])
        file_handler.setLevel(config['level'])
        file_handler.setFormatter(PrintStyleFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    
    # Log initialization to console
    print(f"Logging system initialized - Session: {session_time}")
    print(f"Log directory: {os.path.abspath(log_dir)}")
    print(f"Debug mode: {debug_mode}, Verbose: {verbose}")
    
    return session_time


class TraderLogger:
    """
    Helper class for trader-specific logging with context.
    Only includes methods that are actually used.
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
    
    def log_trade_response(self, trade: Any, response: str = None):
        """Log response to a trade"""
        self.print_style_log('info', f"""=== TRADE RESPONSE ===
Trader: {self.trader_id}
Trade: {trade}
Response: {response}""")
    
    def print_style_log(self, level: str, multi_line_message: str):
        """
        Log a multi-line message in print-style format
        
        Args:
            level: Log level ('debug', 'info', 'warning', 'error')
            multi_line_message: Multi-line string that would normally be printed
        """
        # Get the appropriate logging method
        log_method = getattr(self.logger, level.lower(), self.logger.debug)
        
        # Convert multi-line message to single line with pipe separators
        single_line = multi_line_message.replace('\n', ' | ').replace('  ', ' ').strip()
        
        # Log the single-line message
        log_method(single_line)