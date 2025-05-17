"""
Tests for time series utilities.
"""

from dispatch_benchmark.utils.time_series import time_series_split


def test_walk_forward_split(three_day_prices):
    """Test that the time series split function creates proper walk-forward train/test splits."""
    # Create splits
    n_splits = 3
    splits = time_series_split(
        three_day_prices, n_splits=n_splits, test_size=24, gap=0
    )

    # The function might not be able to create the full number of splits
    # if there's not enough data, so check that we have at least 1 split
    assert len(splits) > 0, "No splits were created"

    # For each split, verify that the train set comes strictly before the test set
    for i, (train_df, test_df) in enumerate(splits):
        # Check that the train and test sets are not empty
        assert len(train_df) > 0, f"Split {i}: Train set is empty"
        assert len(test_df) > 0, f"Split {i}: Test set is empty"

        # Get the max timestamp from the train set and min timestamp from the test set
        train_max_time = train_df["timestamp"].max()
        test_min_time = test_df["timestamp"].min()

        # Check that the train set comes strictly before the test set
        assert (
            train_max_time < test_min_time
        ), f"Split {i}: Train set overlaps with test set"

        # Check that test sets are the requested size
        assert (
            len(test_df) == 24
        ), f"Split {i}: Test set size is {len(test_df)}, expected 24"

        # Check that each split's test set is after the previous split's test set
        if i > 0:
            prev_test_max = splits[i - 1][1]["timestamp"].max()
            assert (
                prev_test_max < test_min_time
            ), f"Split {i}: Test set overlaps with previous test set"

    # Test with a gap
    splits_with_gap = time_series_split(
        three_day_prices, n_splits=2, test_size=24, gap=12
    )

    # Check that we created at least one split with a gap
    assert len(splits_with_gap) > 0, "No splits were created with gap"

    for i, (train_df, test_df) in enumerate(splits_with_gap):
        train_max_time = train_df["timestamp"].max()
        test_min_time = test_df["timestamp"].min()

        # Calculate the difference in rows between train end and test start
        train_max_idx = three_day_prices[
            three_day_prices["timestamp"] == train_max_time
        ].index[0]
        test_min_idx = three_day_prices[
            three_day_prices["timestamp"] == test_min_time
        ].index[0]

        # Check that there's a gap of 12 rows
        assert (
            test_min_idx - train_max_idx - 1 == 12
        ), f"Split {i}: Gap is not 12"
