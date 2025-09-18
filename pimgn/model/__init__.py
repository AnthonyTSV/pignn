"""
PI-MGN model package
"""

from .pimgn import PIMGN
from .blocks import EdgeBlock, NodeBlock, GlobalBlock

__all__ = ["PIMGN", "EdgeBlock", "NodeBlock", "GlobalBlock"]
