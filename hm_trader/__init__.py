"""
Hypothetical-Minds Trading Agent Module

A modular implementation of the TraderLLM_HM class based on the Hypothetical-Minds approach.
Breaks down the complex trading logic into clean, manageable components.

Components:
- core: Main trader class
- llm_controller: LLM API interactions
- hypothesis: Hypothesis generation and evaluation
- memory: Memory management
- trading_strategy: Trading decision logic
"""

from .core import TraderLLM_HM

__all__ = ['TraderLLM_HM']