from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an affine transform followed by
    batch normalization followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: scaling and shift parameters for batch normalization
    - bn_param: running mean and variance

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    affine_out, affine_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_out)
    cache = (affine_cache, bn_cache, relu_cache)
    return out, cache


def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-BN-relu convenience layer
    """
    affine_cache, bn_cache, relu_cache = cache
    dx_relu = relu_backward(dout, relu_cache)
    dx_bn, dgamma, dbeta = batchnorm_backward_alt(dx_relu, bn_cache)
    dx, dw, db = affine_backward(dx_bn, affine_cache)

    return dx, dw, db, dgamma, dbeta


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_conv_relu_pool_forward(x, w1, b1, w2, b2, conv_param, pool_param):
    """
    Convenience layer that performs conv - ReLU - conv - ReLU - pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a1, conv1_cache = conv_forward_fast(x, w1, b1, conv_param)
    s1, relu1_cache = relu_forward(a1)
    a2, conv2_cache = conv_forward_fast(s1, w2, b2, conv_param)
    s2, relu2_cache = relu_forward(a2)
    out, pool_cache = max_pool_forward_naive(s2, pool_param)
    cache = (conv1_cache, relu1_cache, conv2_cache, relu2_cache, pool_cache)
    return out, cache


def conv_relu_conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv - ReLU - conv - ReLU - pool convenience layer
    """
    conv1_cache, relu1_cache, conv2_cache, relu2_cache, pool_cache = cache
    ds2 = max_pool_backward_naive(dout, pool_cache)
    da2 = relu_backward(ds2, relu2_cache)
    ds1, dw2, db2 = conv_backward_fast(da2, conv2_cache)
    da1 = relu_backward(ds1, relu1_cache)
    dx, dw1, db1 = conv_backward_fast(da1, conv1_cache)
    return dx, dw1, db1, dw2, db2








