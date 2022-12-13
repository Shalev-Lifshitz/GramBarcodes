import random
from typing import List, Tuple
import datetime
import os

import cv2
import natsort
import tensorflow as tf
import numpy as np
import torch

from Indexer import Indexer
from Retriever import Retriever
from MetricComputer import MetricComputer
from helpers import create_cross_val_info


class CrossValidator:
    def __init__(self, project_path: str, layers_list: List[str], date: datetime.date = datetime.date.today(),):
        """Initializer for CrossValidator object."""
        # Auto true in TF 2.0
        tf.executing_eagerly()
        random.seed(1)
        np.random.seed(1)

        self.project_path = project_path
        self.layers_list = layers_list
        self.layers_list_str = ','.join([layer[5] for layer in layers_list])
        self.run_name = f'{date.strftime("%Y-%m-%d")}_layers{self.layers_list_str}'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"device: {self.device}\n")

    def run_KIMIA_Path_24(self, perform_retrieval: bool, compute_metrics: bool):
        """Run the retrieval on the Kimia path 24 dataset.

        Note that this is not cross validation, but is included as part of the class for ease of use.
        """
        run_dir_path = os.path.join(self.project_path, 'runs', 'KIMIA_Path_24', self.run_name)
        if not os.path.exists(run_dir_path):
            os.makedirs(run_dir_path)

        print(f"\nRun directory: {run_dir_path} "
              f"\nLayers list: {self.layers_list}")

        if perform_retrieval:
            # Get paths
            data_dir_path \
                = os.path.join(self.project_path, 'data', 'KIMIA_Path_24')
            train_data_dir_path \
                = os.path.join(self.project_path, 'data', 'KIMIA_Path_24', 'Training')
            test_data_dir_path \
                = os.path.join(self.project_path, 'data', 'KIMIA_Path_24', 'Testing')
            train_gram_barcodes_path \
                = os.path.join(data_dir_path, f"train_gram_barcodes_layers[{self.layers_list_str}].npy")
            test_gram_barcodes_path \
                = os.path.join(data_dir_path, f"test_gram_barcodes_layers[{self.layers_list_str}].npy")

            # Instantiate retriever
            retriever = Retriever(self.device)

            # Get barcodes and classes arrays
            train_gram_barcodes = np.load(train_gram_barcodes_path)
            test_gram_barcodes = np.load(test_gram_barcodes_path)

            running_sum_class_sizes_train = retriever.get_running_sum_class_sizes(train_data_dir_path)
            running_sum_class_sizes_test = retriever.get_running_sum_class_sizes(test_data_dir_path)
            train_classes = np.array([retriever.get_class_from_index(running_sum_class_sizes_train, i)
                                      for i in range(train_gram_barcodes.shape[0])])
            test_classes = np.array([retriever.get_class_from_index(running_sum_class_sizes_test, i)
                                     for i in range(test_gram_barcodes.shape[0])])

            # Perform retrieval
            retriever.perform_retrieval(run_dir_path, train_gram_barcodes, test_gram_barcodes,
                                        train_classes, test_classes)

        if compute_metrics:
            metric_computer = MetricComputer('KIMIA_Path_24')
            metrics = metric_computer.compute_metrics(run_dir_path)

            run_info = (f'np & nw & n_totol \\\\'
                        f'\n{metrics[0]} & {metrics[1]} & {metrics[2]}')
            run_info_file = open(os.path.join(run_dir_path, 'run_info.txt'), "w+")
            run_info_file.write(run_info)
            run_info_file.close()

            return metrics

    def index_KIMIA_Path_24(self):
        """Generate gram barcodes for each fold."""
        data_dir_path = os.path.join(self.project_path, 'data', 'KIMIA_Path_24')

        print("-" * 30 + f"\nINDEXING DATASET KIMIA_Path_24 with layers {self.layers_list_str}...")

        train_data_dir_path = os.path.join(data_dir_path, 'Training')
        test_data_dir_path = os.path.join(data_dir_path, 'Testing')

        print("\nINDEXING TRAINING IMAGES...")
        Indexer.index_images(data_dir_path, train_data_dir_path, self.layers_list,
                             f"train_gram_barcodes_layers[{self.layers_list_str}]")
        print("\nINDEXING TESTING IMAGES...")
        Indexer.index_images(data_dir_path, test_data_dir_path, self.layers_list,
                             f"test_gram_barcodes_layers[{self.layers_list_str}]")

    def run_cross_val(self, dataset_name: str,
                      perform_retrieval: bool, compute_metrics: bool, only_first_fold: bool = False):
        """Run cross validation on dataset_name rounds (auto-detect fold number).

        dataset_name should be one of ['colorectal', 'endometrial']
        """
        assert dataset_name in {'colorectal', 'endometrial'}

        data_dir_path = os.path.join(self.project_path, 'data', dataset_name)
        run_dir_path = os.path.join(self.project_path, 'runs', dataset_name, self.run_name)
        rounds_dir_path = os.path.join(data_dir_path, 'folds')

        k = len(os.listdir(rounds_dir_path))  # number of rounds
        metrics_all_rounds = []
        for i in range(k):
            print('-' * 30, f'\nROUND {i}')

            if not only_first_fold or (only_first_fold and i == 0):
                fold_run_dir_path = os.path.join(run_dir_path, f'test_is_fold{i}')
                if not os.path.exists(fold_run_dir_path):
                    os.makedirs(fold_run_dir_path)

                print(f"\nRun directory: {fold_run_dir_path} "
                      f"\nLayers list: {self.layers_list}")

                if perform_retrieval:
                    train_gram_barcodes, test_gram_barcodes, train_classes, test_classes = \
                        self.load_barcodes_for_round(dataset_name, i)

                    retriever = Retriever(self.device)
                    retriever.perform_retrieval(fold_run_dir_path, train_gram_barcodes, test_gram_barcodes,
                                                train_classes, test_classes)

                if compute_metrics:
                    metric_computer = MetricComputer(dataset_name)
                    metrics = metric_computer.compute_metrics(fold_run_dir_path)
                    metrics_all_rounds.append(metrics)

        cross_val_info, metrics_table_row = create_cross_val_info(metrics_all_rounds, k)
        cross_val_info_file = open(os.path.join(run_dir_path, 'cross_val_info.txt'), "w+")
        cross_val_info_file.write(cross_val_info)
        cross_val_info_file.close()

        if compute_metrics:
            return metrics_table_row

    def load_barcodes_for_round(self, dataset_name: str,
                                test_is_fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train and test gram barcodes arrays for cross val round."""
        assert dataset_name in {'colorectal', 'endometrial'}

        folds_dir_path = os.path.join(self.project_path, 'data', dataset_name, 'folds')
        assert os.path.exists(folds_dir_path)

        # Load test gram barcodes
        test_gram_barcodes_path = os.path.join(folds_dir_path, f'fold{test_is_fold}', 'gram_barcodes',
                                               f"gram_barcodes_layers[{self.layers_list_str}].npy")
        test_gram_barcodes = np.load(test_gram_barcodes_path)

        test_classes = np.load(os.path.join(folds_dir_path, f'fold{test_is_fold}', 'fold_classes.npy'))

        # Load train gram barcodes (need to load the rest of the folds and append to one array)
        train_fold_dirs = [fold_dir for fold_dir in natsort.natsorted(os.listdir(folds_dir_path))
                           if fold_dir[:4] == 'fold' and int(fold_dir[4]) != test_is_fold]

        train_gram_barcodes_lst = []
        train_classes_lst = []
        for train_fold_dir in train_fold_dirs:
            train_gram_barcodes_path = os.path.join(folds_dir_path, train_fold_dir, 'gram_barcodes',
                                                    f"gram_barcodes_layers[{self.layers_list_str}].npy")
            train_gram_barcodes = np.load(train_gram_barcodes_path)
            train_gram_barcodes_lst.append(train_gram_barcodes)

            train_classes = np.load(os.path.join(folds_dir_path, train_fold_dir, 'fold_classes.npy'))
            train_classes_lst.append(train_classes)

        train_gram_barcodes = np.vstack(train_gram_barcodes_lst)
        train_classes = np.concatenate(train_classes_lst)

        return train_gram_barcodes, test_gram_barcodes, train_classes, test_classes

    def index_folds(self, dataset_name: str):
        """Generate gram barcodes for each fold for a dataset."""
        folds_dir_path = os.path.join(self.project_path, 'data', dataset_name, 'folds')
        assert os.path.exists(folds_dir_path)

        print(f"\nINDEXING DATASET {dataset_name} with layers {self.layers_list_str}...")

        fold_dirs = natsort.natsorted(os.listdir(folds_dir_path))
        for fold_dir in fold_dirs:
            print("-" * 30 + f"\nINDEXING FOLD {fold_dir}...")

            fold_dir_path = os.path.join(folds_dir_path, fold_dir)
            save_path = os.path.join(fold_dir_path, 'gram_barcodes')
            Indexer.index_images(save_path, fold_dir_path, self.layers_list,
                                 f"gram_barcodes_layers[{self.layers_list_str}]")

    def create_folds_if_none(self, dataset_name: str, num_folds: int = 5):
        """Create folds for a dataset if not already existing.

        dataset_name must be 'colorectal' or 'endometrial'.
        """
        assert dataset_name in {'colorectal', 'endometrial'}
        image_type = 'tiff' if dataset_name == "colorectal" else 'jpg'

        data_dir_path = os.path.join(self.project_path, 'data', dataset_name)
        folds_dir_path = os.path.join(data_dir_path, 'folds')
        if os.path.exists(folds_dir_path):
            return
        else:
            class_dirs = natsort.natsorted(os.listdir(data_dir_path))
            for class_dir in class_dirs:
                print(f"Creating folds for class: {class_dir}")

                path = os.path.join(data_dir_path, class_dir)
                image_names = os.listdir(path)

                images = []
                for image_name in image_names:
                    image = cv2.imread(os.path.join(path, image_name))
                    images.append(image)

                fold_size = len(images) / num_folds  # number of images in each fold of this class

                # Split images list into num_folds random lists with each as close to fold_size as possible
                random.shuffle(images)
                images_per_fold = [images[int(round(fold_size * i)):int(round(fold_size * (i + 1)))]
                                   for i in range(num_folds)]

                for fold in range(num_folds):
                    save_path = os.path.join(data_dir_path, "folds", f"fold{fold}", class_dir)

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    for i in range(len(images_per_fold[fold])):
                        image_save_path = os.path.join(save_path, f"class{class_dir}_fold{fold}_image{i}.{image_type}")
                        cv2.imwrite(image_save_path, images_per_fold[fold][i])

            # Save an array of the class of each image in each fold (in order)
            for fold in range(num_folds):
                fold_classes = []
                class_dirs = natsort.natsorted(class_dirs)
                for i in range(len(class_dirs)):
                    class_dir_path = os.path.join(folds_dir_path, f'fold{fold}', class_dirs[i])
                    class_dir_content = os.listdir(class_dir_path)

                    for _ in class_dir_content:
                        fold_classes.append(i)

                fold_classes = np.asarray(fold_classes)
                np.save(os.path.join(data_dir_path, "folds", f"fold{fold}", 'fold_classes.npy'), fold_classes)
