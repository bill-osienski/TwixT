"""
Pytest configuration and shared fixtures for TwixT tests.
"""

import sys
from pathlib import Path

import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "GPU"))


@pytest.fixture
def game_state():
    """Create a fresh GameState."""
    from game.state import GameState
    return GameState(board_size=24, to_move="red")


@pytest.fixture
def all_edges():
    """Get all canonical knight edges."""
    from game.edge_index import get_all_edges
    return get_all_edges()


@pytest.fixture
def edge_to_idx():
    """Get edge to index mapping."""
    from game.edge_index import get_edge_to_idx
    return get_edge_to_idx()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (>5 seconds)")
    config.addinivalue_line("markers", "oracle: marks tests requiring Node.js")
    config.addinivalue_line("markers", "bridge: marks bridge crossing tests")
