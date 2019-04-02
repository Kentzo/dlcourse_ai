import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    def softmax_row(v):
      exps = np.exp(v - v.max())
      r = exps / exps.sum()
      r[r == 0.0] = np.finfo(float).eps
      return r

    if len(predictions.shape) == 1:
      return softmax_row(predictions)
    else:
      return np.apply_along_axis(softmax_row, 1, predictions)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if len(probs.shape) == 1:
      return -np.log(probs[target_index])
    else:
      return np.negative(np.log(probs[np.arange(probs.shape[0]), target_index])).mean()


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * (W ** 2).sum()
    grad = 2 * reg_strength * W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()

    if len(predictions.shape) == 1:
      dprediction[target_index] -= 1
    else:
      dprediction[np.arange(dprediction.shape[0]), target_index] -= 1
      dprediction /= dprediction.shape[0]

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        Z = np.vectorize(lambda x: x if x > 0 else 0, [X.dtype])(X)
        self.X = X.copy()
        return Z

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return np.where(self.X > 0, d_out, 0.0)

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output) * np.sqrt(2 / n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output) * np.sqrt(2 / n_output))
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        return self.X @ self.W.value + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        self.W.grad += self.X.T @ d_out
        self.B.grad += d_out.sum(axis=0)[np.newaxis, :]
        return d_out @ self.W.value.T

    def params(self):
        return {'W': self.W, 'B': self.B}
