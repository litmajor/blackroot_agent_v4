"""
BLACKROOT AGENT MODULE - All operational agents
Lazy-loaded by kernel on initialization to avoid circular imports
"""

# Lazy imports - loaded only when explicitly imported or by kernel
__all__ = [
    'BaseAgent',
    'ScoutAgent', 
    'InfiltratorAgent',
    'ExfiltratorAgent',
    'StealthManager',
    'LearningAgent',
    'IntelligentOperationsAgent',
]

