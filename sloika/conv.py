from collections import OrderedDict
import numpy as np
import theano as th
import theano.tensor as T
import theano.tensor.signal.pool as tp


def conv_same_1d(X, W, stride=1):
    """1d convolution in "same" mode i.e. output dim = ceil(input dim / stride)

    Constructs a graph for computation of a 1D convolution over a batch of
    samples. The input is padded so that the output length is independent of
    the filter size. This is similar to Theano's border_mode="half" but it works
    for all filter sizes.

    Theano's implementation of convolution does not have a "same" mode, in which
    the output dimensions can be simply calculated as a function of the input
    dimensions and strides, and are independent of the filter size. This
    function hackily fills that void, in a non-idiot-proof fashion.

    :param X: input of shape (time, batch, input_features)
    :param W: a filter of shape (out_features, in_features, winlen)
    :param stride: the rate of downsampling

    Returns:
        A 3D tensor of shape (ceil(input dim / stride), batch, out_features)
    """

    winlen = T.shape(W)[2]
    pad = T.shape_padleft(T.zeros(X.shape[1:]))
    pad_begin = T.repeat(pad, (winlen - 1) // 2, axis=0)
    pad_end = T.repeat(pad, winlen // 2, axis=0)
    X_pad = T.concatenate([pad_begin, X, pad_end], 0)

    conv = T.nnet.conv2d(T.shape_padaxis(X_pad.transpose((1, 2, 0)), 2),
                         T.shape_padaxis(W, 2), subsample=(1, stride),
                         filter_flip=False)
    Y = conv.transpose((3, 0, 1, 2))[:, :, :, 0]

    return Y


def pool_same_1d(X, pool_size, stride):
    """1d max pool in "same" mode i.e. output dim = ceil(input dim / stride)

    Theano's implementation of pool does not have a "same" mode, in which the
    output dimensions can be simply calculated as a function of the input
    dimensions and strides, and are independent of the pool size. This
    function hackily fills that void, in a non-idiot-proof fashion.

    :param X: input of shape (time, batch, features)
    :param size: length of pool
    :param stride: level of downsampling

    Returns:
        3D tensor of shape (ceil(time/stride), batch, features)
    """

    pad = T.shape_padleft(T.zeros(X.shape[1:]))
    pad_begin = T.repeat(pad, (pool_size - 1) // 2, axis=0)
    pad_end = T.repeat(pad, pool_size // 2, axis=0)
    X_pad = T.concatenate([pad_begin, X, pad_end], 0)

    pool = tp.pool_2d(T.shape_padaxis(X_pad.transpose((1, 2, 0)), 2),
                      (1, pool_size), st=(1, stride), ignore_border=True)
    Y = pool.transpose((3, 0, 1, 2))[:, :, :, 0]

    return Y
