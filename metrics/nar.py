import numpy as np
import pandas as pd
from typing import Dict


def normalized_accuracy_range(
    results_df: pd.DataFrame,
    fitz_column: str,
    gt_diag_column: str,
    pred_diag_column: str,
    report_group_accs: bool = False,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Calculate the Normalized Accuracy Range (NAR) from a results DataFrame.
    Note that the FST labels are assumed to be 0-indexed.

    NAR is defined as the range of accuracy values across all protected
    groups, normalized by the mean accuracy.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of the
                                   model's predictions.
                                   This DataFrame should have columns for the
                                   FST label, ground truth diagnosis, and
                                   predicted diagnosis.
        fitz_column (str): Column name in `results_df` that contains the FST
                           labels.
        gt_diag_column (str): Column name in `results_df` that contains the
                              ground truth diagnoses.
        pred_diag_column (str): Column name in `results_df` that contains the
                                predicted diagnoses.
        report_group_accs (bool): Flag to return per-group accuracies.
        eps (float): A small constant for numerical stability.

    Returns:
        Dict with the following keys:
            - "NAR": The Normalized Accuracy Range.
            - "group_accs": Dict of per-group accuracies, if `report_group_accs`
              is set to `True`.
    """

    # Compute per-group accuracies (FSTs 1-6, but 0-indexed).
    unique_fst_groups = results_df[fitz_column].unique()
    group_accs = {}

    for fst in unique_fst_groups:
        fst_mask = results_df[fitz_column] == fst
        group_diag_gt = results_df.loc[fst_mask, gt_diag_column]
        group_diag_pred = results_df.loc[fst_mask, pred_diag_column]

        # Compute per-group accuracies.
        # If there are no cases in this group, set its accuracy to 0.
        if len(group_diag_gt) == 0:
            group_acc = 0.0
        else:
            group_acc = (group_diag_pred == group_diag_gt).mean()

        group_accs[fst] = group_acc

    # Compute NAR
    acc_values = np.array(list(group_accs.values()))
    acc_max, acc_min = acc_values.max(), acc_values.min()
    acc_mean = acc_values.mean()

    nar = (acc_max - acc_min) / (acc_mean + eps)

    if report_group_accs:
        return {
            "NAR": nar,
            "group_accs": group_accs,
        }
    else:
        return {"NAR": nar}
