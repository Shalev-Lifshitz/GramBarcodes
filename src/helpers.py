import statistics


def create_cross_val_info(metrics_all_rounds: list, k: int):
    """Create a text file containing cross validation results, in latex format."""
    if len(metrics_all_rounds) > 1:
        accuracy_all_rounds = [metrics[0] for metrics in metrics_all_rounds]
        sensitivity_all_rounds = [metrics[1] for metrics in metrics_all_rounds]
        specificity_all_rounds = [metrics[2] for metrics in metrics_all_rounds]
        auc_all_rounds = [metrics[3] for metrics in metrics_all_rounds]

        accuracy_mean = round(statistics.mean(accuracy_all_rounds), 2)
        accuracy_std = round(statistics.stdev(accuracy_all_rounds), 2)

        sensitivity_mean = round(statistics.mean(sensitivity_all_rounds), 2)
        sensitivity_std = round(statistics.stdev(sensitivity_all_rounds), 2)

        specificity_mean = round(statistics.mean(specificity_all_rounds), 2)
        specificity_std = round(statistics.stdev(specificity_all_rounds), 2)

        auc_mean = round(statistics.mean(auc_all_rounds), 4)
        auc_std = round(statistics.stdev(auc_all_rounds), 4)

        metrics_table_row = [f"{accuracy_mean} ± {accuracy_std}",
                             f"& {sensitivity_mean} ± {sensitivity_std}",
                             f"& {specificity_mean} ± {specificity_std}",
                             f"& {auc_mean} ± {auc_std}"]

        cross_val_info = (
            f"k (num rounds): {k}"
            f"\n\nAccuracy & Sensitivity & Specificity & AUC \\\\"
            f"\n{metrics_table_row}"
        )
    else:
        cross_val_info = (
            f"k (num rounds): {k}"
            f"\n metrics: \n{metrics_all_rounds}"
        )

    return cross_val_info, metrics_table_row
