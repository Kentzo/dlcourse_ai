from itertools import chain

import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.conv1_layer = (
            ConvolutionalLayer(input_shape[2], conv1_channels, 3, 0),
            ReLULayer(),
            MaxPoolingLayer(4, 1)
        )
        self.conv2_layer = (
            ConvolutionalLayer(conv1_channels, conv2_channels, 3, 0),
            ReLULayer(),
            MaxPoolingLayer(4, 1)
        )
        final_shape = (
            input_shape[0] - 3 + 1 - 4 + 1 - 3 + 1 - 4 + 1,
            input_shape[1] - 3 + 1 - 4 + 1 - 3 + 1 - 4 + 1,
            conv2_channels
        )
        self.output_layer = (
            Flattener(),
            FullyConnectedLayer(np.prod(final_shape), n_output_classes)
        )

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for p in self.params().values():
            p.reset()

        o = self.forward(X)
        loss, loss_grad = softmax_with_cross_entropy(o, y)
        self.backward(loss_grad)
        return loss

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def params(self):
        conv1_params = self.conv1_layer[0].params()
        conv2_params = self.conv2_layer[0].params()
        output_params = self.output_layer[1].params()

        return {
            'c1W': conv1_params['W'],
            'c1B': conv1_params['B'],
            'c2W': conv2_params['W'],
            'c2B': conv2_params['B'],
            'oW': output_params['W'],
            'oB': output_params['B']
        }

    def forward(self, X):
        for l in chain(self.conv1_layer, self.conv2_layer, self.output_layer):
            X = l.forward(X)

        return X

    def backward(self, d_out):
        for l in chain(reversed(self.output_layer), reversed(self.conv2_layer), reversed(self.conv1_layer)):
            d_out = l.backward(d_out)

        return d_out