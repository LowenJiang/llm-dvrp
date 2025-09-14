"""
Utils package for DVRP environment.

This package contains utility modules for:
- sim: Simulation and data generation
- nodify: Network creation and processing
- solver: Route optimization and cost estimation
- user: User acceptance functions
- route_class: Route class implementation (if needed)
"""

# Import main modules for easy access
from . import sim
from . import nodify
from . import solver
from . import user

# Optionally import route_class if it's used
try:
    from . import route_class
except ImportError:
    pass

# Make commonly used functions available at package level
try:
    from .sim import simulation
    from .nodify import create_network
    from .solver import cost_estimator
    from .user import dummy_user
except ImportError as e:
    # If specific functions don't exist, that's okay
    pass

__all__ = ['sim', 'nodify', 'solver', 'user']