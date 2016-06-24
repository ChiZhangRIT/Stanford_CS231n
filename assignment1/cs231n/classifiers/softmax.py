import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_samples = X.shape[0]
    num_classes = W.shape[1]

    for i in xrange(num_samples):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # push constant into the sum to prevent numerically unstable
        scores = np.exp(scores) / np.sum(np.exp(scores))
        loss += -1 * np.log(scores[y[i]])  # sum over all samples
        for j in xrange(num_classes):
            dW[:, j] += (scores[j] - (j == y[i])) * X[i, :]

    loss /= num_samples  # average over all samples
    loss += 0.5 * reg * np.sum(W * W)  # add regularization

    dW /= num_samples
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_samples = X.shape[0]

    scores = X.dot(W)  # shape (N, C)
    scores -= np.max(scores, axis=1).reshape((num_samples, -1))
    scores = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape((num_samples, -1))
    loss = -1 * np.log(scores[np.arange(num_samples), y])  # shape (N, 1)
    loss = np.mean(loss)  # shape scalar
    loss += 0.5 * reg * np.sum(W * W)  # add regularization

    scores[np.arange(num_samples), y] += -1
    dW = X.T.dot(scores)
    dW /= num_samples
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

