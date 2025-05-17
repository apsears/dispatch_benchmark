"""
A simple test to verify that our package is installed correctly.
"""


def test_imports():
    """Test that we can import our package."""
    try:
        import dispatch_benchmark

        assert True
    except ImportError:
        assert False, "Failed to import dispatch_benchmark package"

    try:
        from dispatch_benchmark.utils import time_series_split

        assert True
    except ImportError:
        assert False, "Failed to import time_series_split"
