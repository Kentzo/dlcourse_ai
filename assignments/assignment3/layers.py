import numpy as np


def reshape_as_view(a, shape):
    b = a.view()
    b.shape = shape
    return b


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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

    def reset(self):
        self.grad = np.zeros_like(self.value)


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


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W = Param(np.random.randn(filter_size, filter_size, in_channels, out_channels))
        self.B = Param(np.zeros(out_channels, dtype=np.float))
        self.X = None

        self.padding = padding


    def forward(self, X):
        self.X = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant').copy()
        self.X.flags.writeable = False
        batch_size, height, width, channels = self.X.shape

        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1
        out = np.zeros((batch_size, out_height, out_width, self.out_channels), dtype=np.float)

        for y in range(out_height):
            for x in range(out_width):
                X_region = self.X[:, y:y+self.filter_size, x:x+self.filter_size]
                out[:, y, x, :] = np.tensordot(X_region, self.W.value, axes=3) + self.B.value

        return out


    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        X_grad = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                X_region_slice = (slice(None), slice(y, y + self.filter_size), slice(x, x + self.filter_size), slice(None))
                d_out_region = d_out[:, y, x, :]

                X_grad[X_region_slice] += np.tensordot(d_out_region, self.W.value, axes=([1], [3]))
                self.W.grad += np.tensordot(self.X[X_region_slice], d_out_region, axes=([0], [0]))
                self.B.grad += d_out_region.sum(axis=0)

        if self.padding:
            X_grad = X_grad[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return X_grad

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        batch_size, height, width, channels = X.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        out = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                X_y = y * self.stride
                X_x = x * self.stride
                X_region_slice = (slice(None), slice(X_y, X_y + self.pool_size), slice(X_x, X_x + self.pool_size), slice(None))
                out[:, y, x, :] += np.amax(X[X_region_slice], axis=(1, 2))

        return out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape

        X_grad = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                X_y = y * self.stride
                X_x = x * self.stride

                X_region_slice = (slice(None), slice(X_y, X_y + self.pool_size), slice(X_x, X_x + self.pool_size), slice(None))
                X_region = self.X[X_region_slice].reshape(batch_size, self.pool_size * self.pool_size, channels)

                X_region_argmax = (
                    np.repeat(np.arange(batch_size), channels),
                    *np.unravel_index(X_region.argmax(axis=1).ravel(), (self.pool_size, self.pool_size)),
                    np.tile(np.arange(channels), batch_size)
                )

                X_grad[X_region_slice][X_region_argmax] += d_out[:, y, x, :].ravel()

        return X_grad


    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
