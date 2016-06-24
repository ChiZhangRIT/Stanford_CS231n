import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    #############################################################################
    # TODO: Implement the momentum update formula. Store the updated value in   #
    # the next_w variable. You should also use and update the velocity v.       #
    #############################################################################
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    #############################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of x   #
    # in the next_x variable. Don't forget to update cache value stored in      #
    # config['cache'].                                                          #
    #############################################################################
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dx**2
    next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon'])
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return next_x, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    Reference :
    Diederik Kingma, Jimmy Ba, "Adam: A Method for Stochastic Optimization," arXiv:1412.6980 [cs.LG], 2014
    <http://arxiv.org/abs/1412.6980>

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient, range [0,1).
    - beta2: Decay rate for moving average of second moment of gradient, range [0,1).
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    next_x = None
    #############################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in   #
    # the next_x variable. Don't forget to update the m, v, and t variables     #
    # stored in config.                                                         #
    #############################################################################
    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dx**2
    mb = config['m'] / (1 - config['beta1']**config['t'])
    vb = config['v'] / (1 - config['beta2']**config['t'])
    next_x = x - config['learning_rate'] * mb / (np.sqrt(vb) + config['epsilon'])
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return next_x, config


def adamax(x, dx, config=None):
    """
    Uses the AdaMax update rule,  a variant of Adam based on the infinity norm.

    Reference :
    Diederik Kingma, Jimmy Ba, "Adam: A Method for Stochastic Optimization," arXiv:1412.6980 [cs.LG], 2014
    <http://arxiv.org/abs/1412.6980>

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient, range [0,1).
    - beta2: Decay rate for moving average of second moment of gradient, range [0,1).
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient, 1st momentum vector.
    - u: Exponentially weighted infinity norm.
    - t: Iteration number (time step).
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 2e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('u', np.zeros_like(x))
    config.setdefault('t', 0)

    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
    config['u'] = np.maximum(config['beta2'] * config['u'], np.absolute(dx))
    bias_correct_learning_rate = config['learning_rate'] / (1 - config['beta1']**config['t'])
    next_x = x - bias_correct_learning_rate * config['m'] / (config['u'] + config['epsilon'])

    return next_x, config


def nesterov_momentum(w, dw, config=None):
    """
    Nesterov Momentum: A slightly different version of the SGD momentum update.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    config.setdefault('velocity', np.zeros_like(w))
    v_prep = config['velocity']
    config['velocity'] = config['momentum'] * config['velocity'] - config['learning_rate'] * dw
    next_w = w - config['momentum'] * v_prep + (1 + config['momentum']) * config['velocity']

    return next_w, config


def adagrad(x, dx, config=None):
    """
    Perform the adaptive gradient algorithm.

    Reference:
    John Duchi, Elad Hazan, Yoram Singer, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization," JMLR, 2011.
    <http://jmlr.org/papers/v12/duchi11a.html>

    config format:
    - learning_rate: Scalar learning rate.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: size equal to the size of the gradient, and keeps track of per-parameter sum of squared gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    config['cache'] += dx**2
    next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon'])

    return next_x, config


def adadelta(x, dx, config=None):
    """
    Adadelta is a gradient descent based optimizer that adapts the learning rate per parameter over time.

    Reference:
    Matthew D. Zeiler, "ADADELTA: An Adaptive Learning Rate Method," arXiv:1212.5701 [cs.LG], 2012.
    <http://arxiv.org/abs/1212.5701>

    config format:
    - decay_rate: Scalar decay rate.
    - epsilon: Scalar, RMS regularizer.
    - ms_grad_x: Accumulated mean squared gradient, E[g^2], same size as x.
    - ms_delta_x: Accumulated mean squared Parameter update, E[Delta x^2], same size as x.
    """
    if config is None: config = {}
    config.setdefault('decay_rate', 0.9)
    config.setdefault('epsilon', 1e-6)
    config.setdefault('ms_grad_x', np.zeros_like(x))
    config.setdefault('ms_delta_x', np.zeros_like(x))

    rho = config['decay_rate']
    epsilon = config['epsilon']

    config['ms_grad_x'] = rho * config['ms_grad_x'] + (1 - rho) * dx**2
    delta_x = np.sqrt(config['ms_delta_x'] + epsilon) / np.sqrt(config['ms_grad_x'] + epsilon) * dx
    config['ms_delta_x'] = rho * config['ms_delta_x'] + (1 - rho) * delta_x**2
    next_x = x - delta_x

    return next_x, config


