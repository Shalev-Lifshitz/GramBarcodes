from typing import List, Tuple

import natsort
import numpy as np
import torch
import os


class Retriever:
    def __init__(self, device: torch.device):
        self.device = device

    def perform_retrieval(self, run_dir_path: str, train_gram_barcodes: np.ndarray, test_gram_barcodes: np.ndarray,
                          train_classes: np.ndarray, test_classes: np.ndarray) -> None:
        """Perform retrieval and save query_classes.npy and retrieval_classes.npy

        Args:
            run_dir_path: Path to run dir.
            train_gram_barcodes: Generated gram barcodes for database images.
            test_gram_barcodes: Generated gram barcodes for query images.
            train_classes: Array of all database barcode classes.
            test_classes: Array of all query barcode classes.
        """
        distances = self.compute_distances(train_gram_barcodes, test_gram_barcodes)
        query_classes, retrieval_classes = \
            self.distances_to_classes(distances, train_classes, test_classes)

        np.save(os.path.join(run_dir_path, 'query_classes.npy'), query_classes)
        np.save(os.path.join(run_dir_path, 'retrieval_classes.npy'), retrieval_classes)

    # noinspection PyArgumentList
    def compute_distances(self, train_gram_barcodes: np.ndarray,
                          test_gram_barcodes: np.ndarray) -> np.ndarray:
        """Compute and save distance array, but split array to save VRAM."""
        # Split train_gram_barcodes into parts (vertically) to save VRAM
        num_train = train_gram_barcodes.shape[0]
        train_gram_barcodes_1 = train_gram_barcodes[:num_train // 4]
        train_gram_barcodes_2 = train_gram_barcodes[num_train // 4:num_train * 2 // 4]
        train_gram_barcodes_3 = train_gram_barcodes[num_train * 2 // 4:num_train * 3 // 4]
        train_gram_barcodes_4 = train_gram_barcodes[num_train * 3 // 4:num_train]

        # Save XOR distances in a distances array where i = query test image and j = the comparison
        # train image (each row is the comparison of one test image with all training images).
        # Compute for all parts and then stack
        print("\nGENERATING DISTANCE ARRAY...", end='')
        distances_1 = torch.cdist(torch.Tensor(test_gram_barcodes).to(self.device),
                                  torch.Tensor(train_gram_barcodes_1).to(self.device), p=0).cpu().numpy()
        distances_2 = torch.cdist(torch.Tensor(test_gram_barcodes).to(self.device),
                                  torch.Tensor(train_gram_barcodes_2).to(self.device), p=0).cpu().numpy()
        distances_3 = torch.cdist(torch.Tensor(test_gram_barcodes).to(self.device),
                                  torch.Tensor(train_gram_barcodes_3).to(self.device), p=0).cpu().numpy()
        distances_4 = torch.cdist(torch.Tensor(test_gram_barcodes).to(self.device),
                                  torch.Tensor(train_gram_barcodes_4).to(self.device), p=0).cpu().numpy()
        distances = np.hstack((distances_1, distances_2, distances_3, distances_4))

        assert np.sum(distances) > 0
        print(f"FINISHED"
              f"\nSHAPE: {distances.shape}")

        return distances

    def distances_to_classes(self, distances: np.ndarray,
                             train_classes: np.ndarray, test_classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Load computed distances and compute array of retrieved image classes (ordered by similarity).

        Args:
            distances: Distances array where each element (i, j) is the calculated distance
                from query image i to database image j.
            train_classes: Array of all database barcode classes.
            test_classes: Array of all query barcode classes.
        """
        assert distances.shape[0] == len(test_classes)
        assert distances.shape[1] == len(train_classes)

        # Array of query image classes
        query_classes = test_classes

        # Compute array of indices that would sort each row in ascending order
        # (each row is a sorted_retrievals_single_query of indices that would sort the row)
        retrieval_indices = np.argsort(distances)
        retrieval_classes = np.empty(distances.shape)
        for i in range(distances.shape[0]):
            for j in range(retrieval_indices.shape[1]):
                retrieval_classes[i, j] = train_classes[retrieval_indices[i, j]]

        return query_classes, retrieval_classes

    def get_class_from_index(self, running_sum_class_sizes: List[int], index: int) -> int:
        """Calculate the class of an image by treating each folder in data_dir_path as a
        class, sort the folders, and see which folder the image index belongs to.

        Args:
            running_sum_class_sizes: List where each element, i, is the number of images in class i + number of images in
                all previous classes.
            index: index of an image in database with structure following running_sum_class_sizes (first index is 0).

        Returns:
            int: class of index (first class is 0)
        """
        for i in range(len(running_sum_class_sizes)):
            if i == 0:
                if 0 <= index < running_sum_class_sizes[i]:
                    return i
            else:
                if running_sum_class_sizes[i - 1] <= index < running_sum_class_sizes[i]:
                    return i
        assert False, "No matching index found."

    def get_running_sum_class_sizes(self, data_dir_path: str) -> List[int]:
        """Return list where each element, i, is the number of images in class i + number of images in all previous classes.

        Args:
            data_dir_path: Path to data directory containing class folders.

        Returns:
            List[int]: running_sum_class_sizes list.
        """
        running_sum_class_sizes = []
        class_folders = natsort.natsorted(os.listdir(data_dir_path))
        for i in range(len(class_folders)):
            folder_path = os.path.join(data_dir_path, class_folders[i])
            class_length = len(os.listdir(folder_path))

            if i == 0:
                running_sum_class_sizes.append(class_length)
            else:
                running_sum_class_sizes.append(class_length + running_sum_class_sizes[i - 1])
        return running_sum_class_sizes

