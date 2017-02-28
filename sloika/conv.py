from collections import OrderedDict
import numpy as np
import theano as th
import theano.tensor as T
import theano.tensor.signal.pool as tp


def pad_same(X, size):
    """Pad first dimension in preparation for conv_same_1d or pool_same_1d

    :param X: symbolic tensor to pad
    :param size: window length or pool size or conv or pool op. If size if odd,
        padding is symmetric (size - 1) // 2 elements at the start and end.
        If size is even, padding at the end is greater by 1. These conventions
        ensure that after a 'valid' conv or pool op, the output shape is
        independent of the window length or pool size.
    """
    pad = T.shape_padleft(T.zeros(X.shape[1:]))
    pad_begin = T.repeat(pad, (size - 1) // 2, axis=0)
    pad_end = T.repeat(pad, size // 2, axis=0)
    X_pad = T.concatenate([pad_begin, X, pad_end], 0)
    return X_pad


def bf1t(X):
    """Transpose from [time, batch, features] to [batch, features, 1, time]"""
    return T.shape_padaxis(X.transpose((1, 2, 0)), 2)


def tbf(X):
    """Tranpose from [batch, features, 1, time] to [time, batch, features]"""
    return X.transpose((3, 0, 1, 2)).flatten(3)


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
    X_pad = pad_same(X, winlen)
    conv = T.nnet.conv2d(bf1t(X_pad), T.shape_padaxis(W, 2),
                         subsample=(1, stride), filter_flip=False)
    Y = tbf(conv)

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

    X_pad = pad_same(X, pool_size)
    pool = tp.pool_2d(bf1t(X_pad), (1, pool_size), st=(1, stride),
                      ignore_border=True)
    Y = tbf(pool)

    return Y
