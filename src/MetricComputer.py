import math
import os
import statistics
from typing import Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, auc


class MetricComputer:
    def __init__(self, dataset_name: str):
        """Initializer for metric computer.

        dataset_name must be one of 'KIMIA_Path_24', 'colorectal', 'endometrial'.
        """
        assert dataset_name in {'KIMIA_Path_24', 'colorectal', 'endometrial'}
        self.dataset_name = dataset_name

    def compute_metrics(self, run_dir_path: str, k: int = 3):
        """Return metrics for a retrieval run (depending on which dataset it is for).

        Args:
            run_dir_path: Path to run dir.
            k: Number of images to retrieve.
        """
        if self.dataset_name == 'KIMIA_Path_24':
            metrics = self.kimia_path_24_metrics(run_dir_path, k)
        elif self.dataset_name in {'colorectal', 'endometrial'}:
            metrics = self.colorectal_endometrial_metrics(run_dir_path, k)

        print(f'METRICS: {metrics}')

        # Save an info text file containing
        metrics_info = (
            f"run_dir_path: {run_dir_path} "
            f"\nmetrics: {metrics} ")
        # index_file = open(os.path.join(run_dir_path, f"metrics_info_acc{round(accuracy, 2)}.txt"), "w+")
        index_file = open(os.path.join(run_dir_path, "metrics_info.txt"), "w+")
        index_file.write(metrics_info)
        index_file.close()

        return metrics

    def kimia_path_24_metrics(self, run_dir_path: str, k: int = 3) -> Tuple[float, float, float]:
        """Compute kimiaPath24 metrics.

        Returns np_score, nw_score, and n_total_score.
        """
        # Get arrays with class information of queries and retrievals
        query_classes = np.load(os.path.join(run_dir_path, 'query_classes.npy'))
        retrieval_classes = np.load(os.path.join(run_dir_path, 'retrieval_classes.npy'))

        # Make class predictions for all query images
        predictions = self.get_predictions(retrieval_classes, k)

        # Get scores
        y_true = query_classes
        y_pred = predictions

        cnf_mat = confusion_matrix(y_true, y_pred)
        np_score = accuracy_score(y_true, y_pred) * 100
        nw_score = (cnf_mat.diagonal() / cnf_mat.sum(axis=0)).mean() * 100
        n_total_score = (nw_score * np_score) / 100
        return np_score, nw_score, n_total_score

    def colorectal_endometrial_metrics(self, run_dir_path: str, k: int = 3) -> Tuple[float, float, float, float]:
        """Compute colorectal and endometrial metrics.

        Returns accuracy, sensitivity, specificity, and auc scores.
        """
        # Get arrays with class information of queries and retrievals
        query_classes = np.load(os.path.join(run_dir_path, 'query_classes.npy'))
        retrieval_classes = np.load(os.path.join(run_dir_path, 'retrieval_classes.npy'))

        # Make class predictions for all query images
        predictions = self.get_predictions(retrieval_classes, k)

        # Get confusion matrix
        y_true = query_classes
        y_pred = predictions
        cnf_matrix = confusion_matrix(y_true, y_pred)

        # Get tp, tn, fp, fn values (PER CLASS, each is an array with one element for each class)
        fp = cnf_matrix.sum(axis=0) - cnf_matrix.diagonal()
        fn = cnf_matrix.sum(axis=1) - cnf_matrix.diagonal()
        tp = np.diag(cnf_matrix)
        tn = np.sum(cnf_matrix) - (fp + fn + tp)

        # Get accuracy, sensitivity, specificity
        accuracy = np.sum(tp) / np.sum(cnf_matrix)  # using general formula for accuracy
        assert math.isclose(accuracy, np.sum(y_pred == y_true) / len(y_true))  # sanity check
        sensitivity = np.mean(tp / (tp + fn))
        specificity = np.mean(tn / (tn + fp))

        # Compute AUC (PER CLASS)
        auc_scores = []
        for class_label in np.unique(y_true):
            class_pred_prob = np.sum(retrieval_classes[:, :k] == class_label, axis=1) / k
            class_y_true = y_true == class_label

            fpr, tpr, thresholds = roc_curve(class_y_true, class_pred_prob)
            auc_scores.append(auc(fpr, tpr))
        auc_score = np.mean(auc_scores)

        return float(accuracy) * 100, float(sensitivity) * 100, float(specificity) * 100, float(auc_score)

    def get_predictions(self, retrieval_classes: np.ndarray, k: int = 3):
        predictions = []
        for i in range(len(retrieval_classes)):
            try:
                predicted_class = statistics.mode(retrieval_classes[i, :k])
            except statistics.StatisticsError:
                predicted_class = retrieval_classes[i, 0]

            predictions.append(predicted_class)
        predictions = np.array(predictions)
        return predictions
