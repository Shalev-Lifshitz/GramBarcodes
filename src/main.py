import datetime

from CrossValidator import CrossValidator
import os


def main():
    """Main driver for experiments.

    layers_list shapes = (1, 32, 32, 64), (1, 256, 256, 128), (1, 128, 128, 256), (1, 64, 64, 512), (1, 32, 32, 512)
    Default: ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    """
    general_run_dir_path = os.path.join('runs', f'{datetime.date.today().strftime("%Y-%m-%d")}')
    if not os.path.exists(general_run_dir_path):
        os.makedirs(general_run_dir_path)

    num_to_layer = {
        1: 'block1_conv1',
        2: 'block2_conv1',
        3: 'block3_conv1',
        4: 'block4_conv1',
        5: 'block5_conv1'
    }
    layer_combinations = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 5],
        [2, 3, 5],
        [4, 5],
        [3, 5],
        [2, 5],
        [1, 5],
        [1],
        [2],
        [3],
        [4],
        [5],
    ]

    # For each layer_combination, run cross validation on each of the three datasets
    # (colorectal, endometrial, and KIMIAPath24) using the layers in layer_combination
    # to generate the gram barcodes.
    for layer_combination in layer_combinations:
        layer_names = [num_to_layer[num] for num in layer_combination]
        cross_validator = CrossValidator(os.getcwd(), layer_names)

        # Create cross validation folds (if they do not exist)
        cross_validator.create_folds_if_none('colorectal')
        cross_validator.create_folds_if_none('endometrial')

        # Index (generate gram barcodes for) the images in each fold of each dataset
        cross_validator.index_folds('colorectal')
        cross_validator.index_folds('endometrial')
        cross_validator.index_KIMIA_Path_24()

        # Run cross validation by predicting the class of each test image in each dataset
        # and computing the results. Results are saved in txt file in the run dir.
        # These methods also return a string representing a row of the
        # table found in the paper 'Gram Barcodes for Histopathology Tissue Texture Retrieval'.
        metrics_table_row_colo = cross_validator.run_cross_val('colorectal', True, True)
        metrics_table_row_endo = cross_validator.run_cross_val('endometrial', True, True)
        metrics_kimia = cross_validator.run_KIMIA_Path_24(True, True)

        # Create the table row and print to console
        if (metrics_table_row_colo is not None and metrics_table_row_endo is not None
                and metrics_kimia is not None):
            table_columns = ["Layers", "n_p", "n_w", "n_total", "accuracy (colo)", "sensitivity",
                             "specificity", "auc", "accuracy (endo)",
                             "sensitivity", "specificity", "auc"]
            first_row = ""
            second_row = ""
            i = 0
            for j in range(len(table_columns)):
                if j == 0:
                    s = str(layer_combination)
                elif 1 <= j <= 3:
                    s = str(round(metrics_kimia[j - 1], 2))
                elif 4 <= j <= 7:
                    s = metrics_table_row_colo[j - 4]
                elif 7 <= j:
                    s = metrics_table_row_endo[j - 7]

                width = max(len(table_columns[j]), len(s))
                first_row += table_columns[j] + (" " * (width - len(table_columns[j]))) + " "
                second_row += second_row + (" " * (width - len(s))) + " "
            print(first_row)
            print(second_row)


if __name__ == '__main__':
    main()
