from functools import partial
from sloika import layers, sloika_dtype
from numpy.random import normal as rn


_NBASE = 4
_NFEATURE = 4

def _rn(size, sd):
    return rn(size=size, scale=sd).astype(sloika_dtype)

def _wrap_nanonet(net_layers, kmer, winlen, size, bad_state, sd):
    """ Standard wrappinfg for Nanonet like networks

    :param network: Network to wrap
    :param kmer: target kmer size
    :param winlen: Window size
    :param size: size of hidden layers
    :param bad_state: Add output state for bad events
    :param sd: Scale (SD) or noise to initialise weights

    :returns: a `class`:layer.Layer:
    """
    nstate = _NBASE ** kmer + bad_state
    _prn = partial(_rn, sd=sd)

    win_layer = layers.Window(winlen)
    outlayer = layers.Softmax(size, nstate, init=_prn, has_bias=True)
    return layers.Serial([win_layer] + net_layers + [outlayer])

def mlp(kmer=3, winlen=3, size=64, bad_state=True, sd=0.1, fun=layers.tanh):
    """ Create a simple MultiLayer Perceptron

    :param kmer: target kmer size
    :param winlen: Window size
    :param size: size of hidden layers
    :param bad_state: Add output state for bad events
    :param fun: activation function

    :returns: a `class`:layer.Layer:
    """
    _prn = partial(_rn, sd=sd)
    layer1 = layers.FeedForward(winlen * _NFEATURE, size, init=_prn, has_bias=True, fun=fun)
    layer2 = layers.FeedForward(winlen * _NFEATURE, size, init=_prn, has_bias=True, fun=fun)

    return _wrap_nanonet([layer1, layer2], kmer, winlen, size, bad_state, sd)

def lstm(kmer=3, winlen=3, size=64, bad_state=True, sd=0.1, fun=layers.tanh):
    """ Create a single layer LSTM

    :param kmer: target kmer size
    :param winlen: Window size
    :param size: size of hidden layers
    :param bad_state: Add output state for bad events
    :param fun: activation function

    :returns: a `class`:layer.Layer:
    """
    _prn = partial(_rn, sd=sd)
    fwd1 = layers.Lstm(winlen * _NFEATURE, size, init=_prn, has_bias=True, has_peep=True, fun=fun)
    bwd1 = layers.Lstm(winlen * _NFEATURE, size, init=_prn, has_bias=True, has_peep=True, fun=fun)
    layer1 = layers.birnn(fwd1, bwd1)

    return _wrap_nanonet([layer1], kmer, winlen, size, bad_state, sd)

def nanonet(kmer=3, winlen=3, size=64, bad_state=True, sd=0.1, fun=layers.tanh):
    """ Create standard Nanonet

    :param kmer: target kmer size
    :param winlen: Window size
    :param size: size of hidden layers
    :param bad_state: Add output state for bad events
    :param fun: activation function

    :returns: a `class`:layer.Layer:
    """
    _prn = partial(_rn, sd=sd)
    fwd1 = layers.Lstm(winlen * _NFEATURE, size, init=_prn, has_bias=True, has_peep=True, fun=fun)
    bwd1 = layers.Lstm(winlen * _NFEATURE, size, init=_prn, has_bias=True, has_peep=True, fun=fun)
    layer1 = layers.birnn(fwd1, bwd1)

    layer2 = layers.FeedForward(2 * size, size, has_bias=True, fun=fun)

    fwd3 = layers.Lstm(size, size, init=_prn, has_bias=True, has_peep=True, fun=fun)
    bwd3 = layers.Lstm(size, size, init=_prn, has_bias=True, has_peep=True, fun=fun)
    layer3 = layers.birnn(fwd3, bwd3)

    layer4 = layers.FeedForward(2 * size, size, init=_prn, has_bias=True, fun=fun)

    return _wrap_nanonet([layer1, layer2, layer3, layer4], kmer, winlen, size, bad_state, sd)


def lagged(kmer=3, winlen=3, size=64, bad_state=True, sd=0.1, fun=layers.tanh):
    """ Create unidirectional version of nanonet intended for lagged calling

    :param kmer: target kmer size
    :param winlen: Window size
    :param size: size of hidden layers
    :param bad_state: Add output state for bad events
    :param fun: activation function

    :returns: a `class`:layer.Layer:
    """
    lsize = 2 * size
    _prn = partial(_rn, sd=sd)
    layer1 = layers.Lstm(winlen * _NFEATURE, lsize, init=_prn, has_bias=True, has_peep=True, fun=fun)
    layer2 = layers.FeedForward(lsize, size, init=_prn, has_bias=True, fun=fun)
    layer3 = layers.Lstm(size, lsize, init=_prn, has_bias=True, has_peep=True, fun=fun)
    layer4 = layers.FeedForward(lsize, size, init=_prn, has_bias=True, fun=fun)

    return _wrap_nanonet([layer1, layer2, layer3, layer4], kmer, winlen, size, bad_state, sd)

def transducer(winlen=3, size=64, bad_state=True, sd=0.1, fun=layers.tanh, convolution=False):
    """ Create standard Nanonet

    :param winlen: Window size
    :param size: size of hidden layers
    :param bad_state: Add output state for bad events
    :param fun: activation function
    :param convolution: Do a more general convolution rather than windowing

    :returns: a `class`:layer.Layer:
    """
    _prn = partial(_rn, sd=sd)
    nstate = _NBASE + 1 + bad_state

    if convolution:
        inlayer = layers.Convolution(_NFEATURE, _NFEATURE * winlen, winlen, init=_prn)
    else:
        inlayer = layers.Window(winlen)

    fwd1 = layers.Gru(winlen * _NFEATURE, size, init=_prn, has_bias=True, fun=fun)
    bwd1 = layers.Gru(winlen * _NFEATURE, size, init=_prn, has_bias=True, fun=fun)
    layer1 = layers.birnn(fwd1, bwd1)

    layer2 = layers.FeedForward(2 * size, size, has_bias=True, fun=fun)

    fwd3 = layers.Gru(size, size, init=_prn, has_bias=True, fun=fun)
    bwd3 = layers.Gru(size, size, init=_prn, has_bias=True, fun=fun)
    layer3 = layers.birnn(fwd3, bwd3)

    layer4 = layers.FeedForward(2 * size, size, init=_prn, has_bias=True, fun=fun)

    outlayer = layers.Softmax(size, nstate, init=_prn, has_bias=True)

    return layers.Serial([inlayer, layer1, layer2, layer3, layer4, outlayer])


def lagged_transducer(winlen=3, size=64, bad_state=True, sd=0.1, fun=layers.tanh):
    """ Create unidirectional version of nanonet intended for lagged calling

    :param winlen: Window size
    :param size: size of hidden layers
    :param bad_state: Add output state for bad events
    :param fun: activation function

    :returns: a `class`:layer.Layer:
    """
    lsize = 2 * size
    _prn = partial(_rn, sd=sd)
    nstate = _NBASE + 1 + bad_state

    inlayer = layers.Window(winlen)

    layer1 = layers.Lstm(winlen * _NFEATURE, lsize, init=_prn, has_bias=True, has_peep=True, fun=fun)
    layer2 = layers.FeedForward(lsize, size, init=_prn, has_bias=True, fun=fun)
    layer3 = layers.Gru(size, lsize, init=_prn, has_bias=True, fun=fun)
    layer4 = layers.FeedForward(lsize, size, init=_prn, has_bias=True, fun=fun)

    outlayer = layers.Softmax(size, nstate, init=_prn, has_bias=True)

    return layers.Serial([inlayer, layer1, layer2, layer3, layer4, outlayer])
