from tensorflow.python.keras import models
import tensorflow as tf
import numpy as np
import cv2


class GramBarcodeGenerator:
    def __init__(self, layers_list):
        self.model = self.init_model(layers_list)

    def generate(self, image):
        """Return gram barcode for an image."""
        # Preprocess image (resize, expand batch dim, apply VGG norms)
        image = self.prepare_image(image)
        image_tensor = tf.keras.applications.vgg19.preprocess_input(image)

        # Output of model is feature maps of selected layers (4 dims)
        layers_feature_maps = self.model(image_tensor)

        # Calculate gram barcode for each layers' feature maps (tensor)
        # i keeps track of the layer number
        for tensor, i in zip(layers_feature_maps, range(len(layers_feature_maps))):
            # Convert tensor from tf to numpy array and remove batch dim
            tensor = np.asarray(tensor)
            if len(tensor.shape) == 4:
                assert tensor.shape[0] == 1
                # remove batch dim
                tensor = tensor[0]

            tensor_reshaped = tensor.reshape((tensor.shape[0] * tensor.shape[1], tensor.shape[2]))
            gram_matrix = np.matmul(tensor_reshaped.T, tensor_reshaped)

            # Remove top triangle of matrix and vectorize (top triangle is redundant)
            gram_vector = gram_matrix[np.tril_indices(gram_matrix.shape[0])]

            # Binarize gram_vectror to create gram barcode for current layer
            layer_barcode = self.binarize_global(gram_vector)

            if i == 0:
                # If 1 layer is specified, only one loop will run and gram_barcode = layer_barcode
                gram_barcode = layer_barcode
            else:
                # If more than 1 layer is specified, subsequent layer barcodes loops will concatenate
                gram_barcode = np.concatenate((gram_barcode, layer_barcode))
        return gram_barcode

    @staticmethod
    def binarize_global(vector):
        """Binarize a vector using global mean binarization.

        Values >= median of vector are True and values < median are False.
        """
        return True * (vector >= np.median(vector))

    @staticmethod
    def binarize_min_max(vector):
        """Binarize vector with min-max binarization for each row.

        Returns a binary vector 1 element shorter than the passed vector.
        """
        binary_vector = np.empty(len(vector) - 1)
        for i in range(len(vector)):
            if i != len(vector) - 1 and vector[i] < vector[i + 1]:
                # Increase means value of 1
                binary_vector[i] = True
            elif i != len(vector) - 1 and vector[i] > vector[i + 1]:
                # Increase means value of 0
                binary_vector[i] = False
        return binary_vector

    # TODO: Better, do I even need to resize the images down to 512?
    @staticmethod
    def prepare_image(image):
        """Resizes image to VGG19 input dimensions and adds batch dim.

        Args:
            image:

        Returns:

        """
        assert len(image.shape) == 3

        # Change the image size to have maximum height or width of 512.
        # VGG can only read maximum 512 in size.
        # This code keeps the same aspect ration of the image (TODO: BETTER WITHOUT KEEPING ASPECT RATIO?)
        if image.shape[0] > image.shape[1]:
            height = 512
            width = round(image.shape[1] * (512 / image.shape[0]))
        elif image.shape[1] > image.shape[0]:
            height = round(image.shape[0] * (512 / image.shape[1]))
            width = 512
        elif image.shape[0] == image.shape[1]:
            width, height = 512, 512

        # Bilinear interpolation
        image_resized = cv2.resize(image, (width, height))
        # Add a new batch dimension.
        image_expanded = np.expand_dims(image_resized, axis=0)
        return image_expanded

    @staticmethod
    def init_model(layers_list):
        """Creates VGG19 model that outputs specified layers.

        layers_list is a list of VGG19 layer names.
        The returned model will output the activated feature maps of these layers.
        """
        # Creating the VGG19 model
        VGG = tf.keras.applications.vgg19.VGG19(include_top=False)
        # Grabbing the style layers from the model
        layers = [VGG.get_layer(layer).output for layer in layers_list]
        # Defining a new model with the same input and layers of interest as output
        # (so that we can grab the feature maps needed to compute loss)
        return models.Model(VGG.input, layers)
