"""
A simple test to verify that our package is installed correctly.
"""

def test_imports():
    """Test that we can import our package."""
    try:
        import virtual_energy
        assert True
    except ImportError:
        assert False, "Failed to import virtual_energy package"
        
    try:
        from virtual_energy.utils import time_series_split
        assert True
    except ImportError:
        assert False, "Failed to import time_series_split" 