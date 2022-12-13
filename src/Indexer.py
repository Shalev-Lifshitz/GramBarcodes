from GramBarcodeGenerator import GramBarcodeGenerator
import numpy as np
import natsort
import time
import cv2
import os


class Indexer:
    @staticmethod
    def index_images(save_path, data_dir_path, layers_list, barcodes_filename):
        """Generate the gram barcode for each image. Method calculates list of barcodes and
        then coverts the list to an array."""
        tik_indexing = time.time()

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Init GramBarcodeGenerator outside of loop to avoid saving a model to memory each loop
        gram_barcode_generator = GramBarcodeGenerator(layers_list)

        class_dirs = natsort.natsorted(os.listdir(data_dir_path))
        if 'gram_barcodes' in class_dirs:
            class_dirs.remove('gram_barcodes')
        if 'fold_classes.npy' in class_dirs:
            class_dirs.remove('fold_classes.npy')

        # Calculate list of gram barcodes. Each element is the image's gram barcode
        gram_barcodes_lst = []
        for i in range(len(class_dirs)):
            dir_path = os.path.join(data_dir_path, class_dirs[i])
            dir_content = natsort.natsorted(os.listdir(dir_path))

            print(f"generating gram barcodes for class dir {i + 1} ({class_dirs[i]})")
            for j in range(len(dir_content)):
                image = cv2.imread(os.path.join(dir_path, dir_content[j]))
                gram_barcode = gram_barcode_generator.generate(image)
                gram_barcodes_lst.append(gram_barcode)

        tok_indexing = time.time()
        indexing_time_taken = tok_indexing - tik_indexing
        print(f"\nINDEXING FINISHED. LIST LENGTH: {len(gram_barcodes_lst)} ")
        print(f"TIME TAKEN: {indexing_time_taken / 60} minutes")

        gram_barcodes_arr = np.asarray(gram_barcodes_lst)
        np.save(os.path.join(save_path, barcodes_filename), gram_barcodes_arr)


