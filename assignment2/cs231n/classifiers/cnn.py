import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
  
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # Layer 1: Conv layer (conv - relu - 2x2 max pool)
        self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)

        # Layer 2: Affine layer (affine - relu)
        self.params['W2'] = weight_scale * np.random.randn(num_filters * input_dim[1] * input_dim[2] / 4, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        # Layer 3: Output layer (affine - softmax)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        X_conv, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        X_conv_col = X_conv.reshape(X.shape[0], -1)
        X_affine, cache_affine = affine_relu_forward(X_conv_col, W2, b2)
        scores, cache_out = affine_forward(X_affine, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

        dX_affine, dW3, db3 = affine_backward(dout, cache_out)
        grads['W3'] = dW3 + self.reg * W3
        grads['b3'] = db3
        dX_conv_col, dW2, db2 = affine_relu_backward(dX_affine, cache_affine)
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2
        dX_conv = dX_conv_col.reshape(X_conv.shape)
        _, dW1, db1 = conv_relu_pool_backward(dX_conv, cache_conv)
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
  
  
class ConvNet(object):
    """
    A convolutional network with the following architecture:

    [conv - relu - conv - relu - pool] x N - [affine - (batch norm) - relu - (dropout)] x M - [affine - softmax]

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, fc_hidden_dims, num_filters, filter_size=3, stride=1, input_dim=(3, 32, 32),
                 num_classes=10, use_batchnorm=False, dropout=0, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: A list of number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - fc_hidden_dims: A list of number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.num_layers = 1 + len(fc_hidden_dims)
        self.stride = stride
        self.filter_size = filter_size
        self.num_filters = num_filters

        """ Initialize weights and biases for the three-layer convolutional network. """
        # Denote Wij and bi as the weights and biases for the ith conv layer in the jth conv-conv-pool unit.
        # Denote W_i and b_i as the weights and biases for the ith fc layer.

        conv_dims = [input_dim[0]] + num_filters
        fc_dims = [num_filters[-1] * input_dim[1] * input_dim[2] / (4 ** (len(num_filters) / 2))] + fc_hidden_dims + [num_classes]
        self.fc_dims = fc_dims
        assert (len(num_filters) % 2 == 0), "Number of filters must be even."

        # [conv - relu - conv - relu - pool] layers
        for i in xrange(len(num_filters) / 2):
            self.params['W1'+str(i+1)] = weight_scale * np.random.randn(num_filters[2 * i], conv_dims[2 * i], filter_size, filter_size)
            self.params['b1'+str(i+1)] = np.zeros(num_filters[2 * i])
            self.params['W2'+str(i+1)] = weight_scale * np.random.randn(num_filters[2 * i + 1], conv_dims[2 * i + 1], filter_size, filter_size)
            self.params['b2'+str(i+1)] = np.zeros(num_filters[2 * i + 1])

        # Affine layers
        for i in xrange(len(fc_dims) - 1):
            self.params['W_'+str(i+1)] = weight_scale * np.random.randn(fc_dims[i], fc_dims[i+1])
            self.params['b_'+str(i+1)] = np.zeros(fc_dims[i+1])
        """ ######################################################################## """

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        """ Implementing the forward pass for the convolutional network. """
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.filter_size
        conv_param = {'stride': self.stride, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        cache = {}

        # [conv - relu - conv - relu - pool] layers
        for i in xrange(len(self.num_filters) / 2):
            W1 = self.params['W1'+str(i+1)]
            b1 = self.params['b1'+str(i+1)]
            W2 = self.params['W2'+str(i+1)]
            b2 = self.params['b2'+str(i+1)]
            X, cache['conv'+str(i+1)] = conv_relu_conv_relu_pool_forward(X, W1, b1, W2, b2, conv_param, pool_param)

        copy_shape = X.shape
        X.shape = (X.shape[0], -1)

        # Affine layers
        for i in xrange(len(self.fc_dims) - 2):
            W = self.params['W_'+str(i+1)]
            b = self.params['b_'+str(i+1)]
            X, cache['fc'+str(i+1)] = affine_relu_forward(X, W, b)

        W = self.params['W_'+str(len(self.fc_dims)-1)]
        b = self.params['b_'+str(len(self.fc_dims)-1)]
        scores, cache['fc'+str(len(self.fc_dims)-1)] = affine_forward(X, W, b)
        """ ############################################################ """

        if mode == 'test':
            return scores

        loss, grads = 0, {}

        """ Implementing the backward pass for the three-layer convolutional net"""
        # storing the loss and gradients in the loss and grads variables.

        # Hinge loss
        loss, dout = softmax_loss(scores, y)
        # Add regularization
        for i in xrange(len(self.num_filters) / 2):
            W1 = self.params['W1'+str(i+1)]
            W2 = self.params['W2'+str(i+1)]
            loss += 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        for i in xrange(len(self.fc_dims) - 2):
            W = self.params['W_'+str(i+1)]
            loss += 0.5 * self.reg * np.sum(np.square(W))

        # Backprop from the output layer
        dX, dW, db = affine_backward(dout, cache['fc'+str(len(self.fc_dims)-1)])
        grads['W_'+str(len(self.fc_dims)-1)] = dW + self.reg * self.params['W_'+str(len(self.fc_dims)-1)]
        grads['b_'+str(len(self.fc_dims)-1)] = db

        # Backprop through affine layers
        for i in reversed(xrange(1, len(self.fc_dims)-1)):
            dX, dW, db = affine_relu_backward(dX, cache['fc'+str(i)])
            grads['W_'+str(i)] = dW + self.reg * self.params['W_'+str(i)]
            grads['b_'+str(i)] = db

        # Backprop though conv-conv-pool layers
        dX.shape = copy_shape
        for i in reversed(xrange(len(self.num_filters) / 2)):
            dX, dW1, db1, dW2, db2 = conv_relu_conv_relu_pool_backward(dX, cache['conv'+str(i+1)])
            grads['W1'+str(i+1)] = dW1 + self.reg * self.params['W1'+str(i+1)]
            grads['b1'+str(i+1)] = db1
            grads['W2'+str(i+1)] = dW2 + self.reg * self.params['W2'+str(i+1)]
            grads['b2'+str(i+1)] = db2

        """ ################################################################### """

        return loss, grads
