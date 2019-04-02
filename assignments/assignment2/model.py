import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg

        self.hidden_layer = (FullyConnectedLayer(n_input, hidden_layer_size), ReLULayer())
        self.output_layer = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for p in self.params().values():
            p.grad = np.zeros_like(p.value)

        h0 = self.hidden_layer[0].forward(X)
        h1 = self.hidden_layer[1].forward(h0)
        o = self.output_layer.forward(h1)

        loss_unreg, loss_unreg_grad = softmax_with_cross_entropy(o, y)

        o_grad = self.output_layer.backward(loss_unreg_grad)
        h1_grad = self.hidden_layer[1].backward(o_grad)
        h0_grad = self.hidden_layer[0].backward(h1_grad)

        loss_reg = 0
        for p in self.params().values():
            p_reg, p_reg_grad = l2_regularization(p.value, self.reg)
            p.grad += p_reg_grad
            loss_reg += p_reg

        return loss_unreg + loss_reg

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        h0 = self.hidden_layer[0].forward(X)
        h1 = self.hidden_layer[1].forward(h0)
        o = self.output_layer.forward(h1)
        return np.argmax(o, axis=1)

    def params(self):
        hidden_params = self.hidden_layer[0].params()
        output_params = self.output_layer.params()

        return {
            'hW': hidden_params['W'],
            'hB': hidden_params['B'],
            'oW': output_params['W'],
            'oB': output_params['B']
        }
