"""
Smoke test for time series splitting functionality.
"""

from dispatch_benchmark.utils.time_series import time_series_split


def test_walk_forward_split(three_day_prices):
    """
    Test that walk-forward time series splits work correctly on 3 days of data.
    Verifies that each train fold ends strictly before its corresponding test fold.
    """
    # Use 3 days of sample data
    df = three_day_prices.copy()

    # Apply time series split with 3 folds
    splits = time_series_split(df, n_splits=3, test_size=0.2, gap=0)

    # Verify we got 3 splits
    assert len(splits) == 3, f"Expected 3 splits, got {len(splits)}"

    # For each split, verify:
    # 1. Train and test sets are not empty
    # 2. Train set ends strictly before test set begins
    for i, (train, test) in enumerate(splits):
        # Check that both train and test are not empty
        assert len(train) > 0, f"Split {i+1}: Train set is empty"
        assert len(test) > 0, f"Split {i+1}: Test set is empty"

        # Get the last timestamp in train and first timestamp in test
        train_end = train["timestamp"].max()
        test_start = test["timestamp"].min()

        # Verify train ends strictly before test begins
        assert (
            train_end < test_start
        ), f"Split {i+1}: Train end ({train_end}) not strictly before test start ({test_start})"

        # Verify there's no overlap between train and test
        train_timestamps = set(train["timestamp"])
        test_timestamps = set(test["timestamp"])
        assert (
            len(train_timestamps.intersection(test_timestamps)) == 0
        ), f"Split {i+1}: Train and test sets have overlapping timestamps"

    # Additional check: Verify that earlier splits use less data than later splits
    # This ensures the walk-forward validation is expanding the training set
    train_sizes = [len(train) for train, _ in splits]
    for i in range(1, len(train_sizes)):
        assert (
            train_sizes[i] > train_sizes[i - 1]
        ), f"Split {i+1} training set ({train_sizes[i]}) should be larger than split {i} ({train_sizes[i-1]})"
