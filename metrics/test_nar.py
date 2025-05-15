import pytest
import pandas as pd
import numpy as np
from metrics.nar import normalized_accuracy_range


def test_nar_perfect_fairness():
    """
    Test that the NAR is 0.0 for perfect fairness.
    """
    perfect_df = pd.DataFrame(
        {
            "gt_diag": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "pred_diag": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "fitz": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        }
    )
    nar_perfect = normalized_accuracy_range(
        results_df=perfect_df,
        fitz_column="fitz",
        gt_diag_column="gt_diag",
        pred_diag_column="pred_diag",
        report_group_accs=True,
    )
    assert nar_perfect["NAR"] == 0.0, "NAR should be 0.0 for perfect fairness."
    assert nar_perfect["group_accs"] == {0: 1.0, 1: 1.0, 2: 1.0}, (
        "All groups should have 100% accuracy."
    )


def test_nar_skewed():
    """
    Test that the NAR is greater than 0.0 for skewed data.
    """
    skewed_df = pd.DataFrame(
        {
            "gt_diag": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            "pred_diag": [0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "fitz": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        }
    )
    nar_skewed = normalized_accuracy_range(
        results_df=skewed_df,
        fitz_column="fitz",
        gt_diag_column="gt_diag",
        pred_diag_column="pred_diag",
        report_group_accs=True,
    )
    assert nar_skewed["NAR"] > 0.0, (
        "NAR should be greater than 0.0 for skewed data."
    )
    assert nar_skewed["group_accs"][0] == 1.0, (
        "Group 0 should have 100% accuracy."
    )
    assert all(
        acc == 0.0 for fst, acc in nar_skewed["group_accs"].items() if fst != 0
    ), "Other groups apart from 0 should have 0% accuracy."


def test_nar_realistic():
    """
    Test that the NAR is equal to the manually computed NAR for realistic 
    data.
    """
    realistic_df = pd.DataFrame(
        {
            "gt_diag": [0] * 10
            + [1] * 10
            + [2] * 10
            + [3] * 10
            + [4] * 10
            + [5] * 10,
            "pred_diag": [0] * 8
            + [1] * 2
            + [1] * 7
            + [0] * 3
            + [2] * 6
            + [0] * 4
            + [3] * 9
            + [1] * 1
            + [4] * 5
            + [0] * 5
            + [5] * 10,
            "fitz": [0] * 10
            + [1] * 10
            + [2] * 10
            + [3] * 10
            + [4] * 10
            + [5] * 10,
        }
    )
    nar_realistic = normalized_accuracy_range(
        results_df=realistic_df,
        fitz_column="fitz",
        gt_diag_column="gt_diag",
        pred_diag_column="pred_diag",
        report_group_accs=True,
    )
    acc_values = np.array(list(nar_realistic["group_accs"].values()))
    acc_max, acc_min = acc_values.max(), acc_values.min()
    acc_mean = acc_values.mean()
    nar_manual = (acc_max - acc_min) / (acc_mean + 1e-6)
    assert np.isclose(nar_realistic["NAR"], nar_manual), (
        "NAR should be equal to the manually computed NAR."
    )


def test_nar_random():
    """
    Test that the NAR is greater than 0.0 for random data.
    """
    num_samples = 1000
    random_df = pd.DataFrame(
        {
            "gt_diag": np.random.randint(0, 114, size=num_samples),
            "pred_diag": np.random.randint(0, 114, size=num_samples),
            "fitz": np.random.randint(0, 6, size=num_samples),
        }
    )
    random_nar = normalized_accuracy_range(
        results_df=random_df,
        fitz_column="fitz",
        gt_diag_column="gt_diag",
        pred_diag_column="pred_diag",
        report_group_accs=False,
        eps=0.0,
    )
    assert random_nar["NAR"] > 0.0, (
        "NAR should be greater than 0.0 for random data."
    )
