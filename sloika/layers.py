import abc
from collections import OrderedDict
import theano as th
import theano.tensor as T
import numpy as np

from sloika import activation, sloika_dtype

"""  Convention: inMat row major (C ordering) as (time, batch, state)
"""
_NBASE = 4
_NSTEP = _NBASE
_NSKIP = _NBASE * _NBASE
_FORGET_BIAS = 2.0
_INDENT = ' ' * 4

def zeros(size):
    return np.zeros(size, dtype=sloika_dtype)

def _extract(x, shape=None):
    xv = x.get_value()
    if shape is not None:
        xv = xv.reshape(shape)
    return xv.tolist()


class Layer(object):
    __metaclass__ = abc.ABCMeta

    def compile(self):
        x = T.tensor3()
        return th.function([th.In(x, borrow=True)], th.Out(self.run(x), borrow=True))

    @abc.abstractmethod
    def params(self):
        """ a list of network parameters
        """
        return

    @abc.abstractmethod
    def json(self, params):
        """ emit json string describing layer
        """
        return

    @abc.abstractmethod
    def set_params(self, values):
        """ Set parameters from a dictionary of values
        """
        return

    @abc.abstractmethod
    def run(self, inMat):
        """  Run network layer
        """
        return


class RNN(Layer):

    @abc.abstractmethod
    def step(self, in_vec, state):
        """ A single step along the RNN
        :param in_vec: Input to node
        :param state: Hidden state from previous node
        """
        return

    def run(self, inMat):
        nbatch = T.shape(inMat)[1]
        out, _ = th.scan(self.step, sequences=inMat, outputs_info=T.zeros((nbatch, self.size)))
        return out


class Identity(Layer):
    def __init__(self):
        pass

    def params(self):
        return []

    def json(self):
        return {'type' : "identity"}

    def set_params(self, values):
        return

    def run(self, inMat):
        return inMat


class FeedForward(Layer):
    """  Basic feedforward layer
         out = f( inMat W + b )

    :params insize: Size of input to layer
    :params size: Layer size
    :params init: function to initialise tensors with
    :params has_bias: Whether layer has bias
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh):
        self.has_bias = has_bias
        self.b = th.shared(has_bias * init(size))
        self.W = th.shared(init((size, insize)) / np.sqrt(size + insize))
        self.insize = insize
        self.size = size
        self.fun = fun

    def params(self):
        return [self.W, self.b] if self.has_bias else [self.W]

    def json(self, params=False):
        res = OrderedDict([('type', "feed-forward"),
                           ('activation', self.fun.func_name),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W', _extract(self.W)),
                                         ('b', _extract(self.b))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape[0] == self.size
            self.b = th.shared(values['b'])
        assert values['W'].shape == (self.size, self.insize)
        self.W = th.shared(values['W'])

    def run(self, inMat):
        return self.fun(T.tensordot(inMat, self.W, axes=(2, 1)) + self.b)


class Studentise(Layer):
    """ Normal all features in batch

    :param epsilon: Stabilsation layer
    """
    def __init__(self, epsilion=1e-4):
        self.epsilion = epsilion

    def params(self):
        return []

    def json(self, params=False):
        return {'type' : "studentise"}

    def set_params(self, values):
        return

    def run(self, inMat):
        m = T.shape_padleft(T.mean(inMat, axis=(0, 1)), n_ones=2)
        v = T.shape_padleft(T.var(inMat, axis=(0, 1)), n_ones=2)
        return (inMat - m) / T.sqrt(v + self.epsilion)


class Softmax(Layer):
    """  Softmax layer
         tmp = exp( inmat W + b )
         out = row_normalise( tmp )

    :params insize: Size of input to layer
    :params size: Layer size
    :params init: function to initialise tensors with
    :params has_bias: Whether layer has bias
    """
    def __init__(self, insize, size, init=zeros, has_bias=False):
        self.has_bias = has_bias
        self.b = th.shared(has_bias * init(size))
        self.W = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self.insize = insize
        self.size = size

    def params(self):
        return [self.W, self.b] if self.has_bias else [self.W]

    def json(self, params=False):
        res = OrderedDict([('type', "softmax"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W', _extract(self.W)),
                                         ('b', _extract(self.b))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape[0] == self.size
            self.b = th.shared(values['b'])
        assert values['W'].shape == (self.size, self.insize)
        self.W = th.shared(values['W'])


    def run(self, inMat):
        tmp =  T.tensordot(inMat, self.W, axes=(2,1)) + self.b
        out, _ = th.map(T.nnet.softmax, sequences=tmp)
        return out


class SoftmaxOld(Layer):
    """  Softmax layer
         tmp = exp( inmat W + b )
         out = row_normalise( tmp )

    :params insize: Size of input to layer
    :params size: Layer size
    :params init: function to initialise tensors with
    :params has_bias: Whether layer has bias
    """
    def __init__(self, insize, size, init=zeros, has_bias=False):
        self.has_bias = has_bias
        self.b = th.shared(has_bias * init(size))
        self.W = th.shared(init((size, insize)) / np.sqrt(size + insize))
        self.insize = insize
        self.size = size

    def params(self):
        return [self.W, self.b] if self.has_bias else [self.W]

    def json(self, params=False):
        res = OrderedDict([('type', "softmax_old"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])

        if params:
            res['params'] = OrderedDict([('W', _extract(self.W)),
                                         ('b', _extract(self.b))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape[0] == self.size
            self.b = th.shared(values['b'])
        assert values['W'].shape == (self.size, self.insize)
        self.W = th.shared(values['W'])

    def run(self, inMat):
        tmp =  T.tensordot(inMat, self.W, axes=(2,1)) + self.b
        m = T.shape_padright(T.max(tmp, axis=2))
        out = T.exp(tmp - m)
        rowsum = T.sum(out, axis=2)
        return out / T.shape_padright(rowsum)


class Window(Layer):
    """  Create a sliding window over input

    :param w: Size of window
    """
    def __init__(self, w):
        assert w > 0, "Window size must be positive"
        assert w % 2 == 1, 'Window size should be odd'
        self.w = w

    def params(self):
        return []

    def json(self, params=False):
        res = OrderedDict([('type', "window")])
        if params:
            res['params'] = OrderedDict([('w', self.w)])

    def set_params(self, values):
        return

    def run(self, inMat):
        ntime, nbatch, nfeatures = T.shape(inMat)
        zeros = T.zeros((self.w // 2, nbatch, nfeatures))
        padMat = T.concatenate([zeros, inMat, zeros], axis=0)
        tmp = T.concatenate([padMat[i : 1 + i - self.w] for i in xrange(self.w - 1)], axis=2)
        return T.concatenate([tmp, padMat[self.w - 1 :]], axis=2)


class Convolution(Layer):
    """ Create a 1D convolution over input

    :params insize: Size of input to layer
    :params size: Layer size (number of filters)
    :param w: Size of convolution
    """
    def __init__(self, insize, size, w, init=zeros, fun=activation.tanh):
        assert size > 0, "Size (number of filters) must be positive"
        assert w > 0, "Window size must be positive"
        self.w = w
        self.flt = th.shared(init((size, insize, 1, w)) / np.sqrt(w * insize + size))
        self.insize = insize
        self.size = size
        self.fun = fun

    def params(self):
        return [self.flt]

    def json(self, params=False):
        res = OrderedDict([('type', "convolution"),
                           ('activation', self.fun.func_name),
                           ('size', self.size),
                           ('insize', self.insize)])
        if params:
            res['params'] = OrderedDict([('w', self.w),
                                         ('filter', _extract(self.flt))])
        return res

    def set_params(self, values):
        assert values['flt'].shape == (self.size, self.insize, 1, self.w)
        self.flt = th.shared(values['flt'])

    def run(self, inMat):
        # Input to convolution is (batch x channels x row x column)
        ntime, nbatch, nfeatures = T.shape(inMat)
        inMatT = T.shape_padaxis(inMat.transpose((1, 2, 0)), axis=2)
        outMat = T.nnet.conv2d(inMatT, filters=self.flt, border_mode='half',
                               filter_shape=(self.size, self.insize, 1, self.w))
        # Output of convolution is (batch x filters x row x col)

        outMat = outMat.transpose((3, 0, 1, 2))
        outMat = outMat.reshape((ntime, nbatch, self.size))
        return self.fun(outMat)


class Recurrent(RNN):
    """ A simple recurrent layer
        Step:  state_new = fun( [state_old, input_new] W + b )
               output_new = state_new

    :params insize: Size of input to layer
    :params size: Layer size
    :params init: function to initialise tensors with
    :params has_bias: Whether layer has bias
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh):
        self.has_bias = has_bias
        self.b = th.shared(has_bias * init(size))
        self.iW = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((size, size)) / np.sqrt(size + size))
        self.fun = fun
        self.insize = insize
        self.size = size

    def params(self):
        return [self.iW, self.sW, self.b] if self.has_bias else [self.iW, self.sW]

    def json(self, params=False):
        res = OrderedDict([('type', "recurrent"),
                           ('activation', self.fun.func_name),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW)),
                                         ('sW', _extract(self.sW)),
                                         ('b', _extract(self.b))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape[0] == self.size
            self.b = th.shared(values['b'])
        assert values['iW'].shape == (self.size, self.insize)
        self.iW = th.shared(values['iW'])
        assert values['sW'].shape == (self.size, self.size)
        self.sW = th.shared(values['sW'])

    def step(self, in_vec, in_state):
        iV = T.tensordot(in_vec, self.iW, axes=(1, 1))
        sV = T.tensordot(in_state, self.sW, axes=(1, 1))
        state_out = self.fun(iV + sV + self.b)
        return state_out


class Lstm(RNN):
    """ LSTM layer with peepholes.  Implementation is to be consistent with
    Currennt and may differ from other descriptions of LSTM networks (e.g.
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

    Step:
        v = [ input_new, output_old ]
        Pforget = sigmoid( v W2 + b2 + state * p1)
        Pupdate = sigmoid( v W1 + b1 + state * p0)
        Update  = tanh( v W0 + b0 )
        state_new = state_old * Pforget + Update * Pupdate
        Poutput = sigmoid( v W3 + b3 + state * p2)
        output_new = tanh(state) * Poutput

    :Note: The inputs are arranged to maintain compatibilty it the older version
    of the LSTM layer and several of the processing steps could be optimised out.

    :params insize: Size of input to layer
    :params size: Layer size
    :params init: function to initialise tensors with
    :params has_bias: Whether layer has bias
    :params has_peep: Whether layer has bias
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, insize, size, init=zeros, has_bias=False, has_peep=False,
                 fun=activation.tanh):
        self.size = size
        self.insize = insize
        self.has_bias = has_bias
        self.has_peep = has_peep
        self.fun = fun

        self.b = th.shared(has_bias * (init(4 * size)
                                       + np.repeat([0, 0, _FORGET_BIAS, 0],
                                                   size).astype(sloika_dtype)))
        self.p = th.shared(has_peep * init((3, size)) / np.sqrt(size))
        self.iW = th.shared(init((4 * size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((4 * size, size)) / np.sqrt(size + size))

    def params(self):
        params =  [self.iW, self.sW]
        if self.has_bias:
            params += [self.b]
        if self.has_peep:
            params += [self.p]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "LSTM"),
                           ('activation', self.fun.func_name),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias),
                           ('peep', self.has_peep)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW, (4, self.size, self.insize))),
                                         ('sW', _extract(self.sW, (4, self.size, self.size))),
                                         ('b', _extract(self.b, (4, self.size))),
                                         ('p', _extract(self.p, (3, self.size)))])
        return res


    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape == (4, self.size)
            self.b = th.shared(values['b'].transpose().reshape(-1))
        if self.has_peep:
            assert values['p'].shape == (3, self.size)
            self.p = th.shared(values['p'])
        assert values['iW'].shape == (4, self.size, self.insize)
        self.iW = th.shared(values['iW'].reshape((self.size * 4, self.insize)))
        assert values['sW'].shape == (4, self.size, self.size)
        self.sW = th.shared(values['sW'].reshape((self.size * 4, self.size)))

    def step(self, in_vec, in_state):
        vW = T.tensordot(in_vec, self.iW, axes=(1, 1))
        out_prev = in_state[:,:self.size]
        state = in_state[:,self.size:]
        outW = T.tensordot(out_prev, self.sW, axes=(1, 1))
        sumW = vW + outW  + self.b
        sumW = sumW.reshape((-1, self.size, 4))

        #  Forget gate activation
        out_state = state * activation.sigmoid(sumW[:,:,2] + state * self.p[1])
        #  Update state with input
        out_state += self.fun(sumW[:,:,0]) * activation.sigmoid(sumW[:,:,1] + state * self.p[0])
        #  Output gate activation
        out = self.fun(out_state) * activation.sigmoid(sumW[:,:,3] + out_state * self.p[2])
        return T.concatenate((out, out_state), axis=1)

    def run(self, inMat):
        nbatch = T.shape(inMat)[1]
        out, _ = th.scan(self.step, sequences=inMat,
                         outputs_info=T.zeros((nbatch, 2 * self.size)))
        return out[:,:,:self.size]


class LstmO(RNN):
    """ LSTM layer with peepholes but no output gate.

    Step:
        v = [ input_new, output_old ]
        Pforget = sigmoid( v W2 + b2 + state * p1)
        Pupdate = sigmoid( v W1 + b1 + state * p0)
        Update  = tanh( v W0 + b0 )
        state_new = tanh(state_old * Pforget + Update * Pupdate)

    :params insize: Size of input to layer
    :params size: Layer size
    :params init: function to initialise tensors with
    :params has_bias: Whether layer has bias
    :params has_peep: Whether layer has bias
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, insize, size, init=zeros, has_bias=False, has_peep=False,
                 fun=activation.tanh):
        self.size = size
        self.insize = insize
        self.has_bias = has_bias
        self.has_peep = has_peep
        self.fun = fun

        self.b = th.shared(has_bias * (init(3 * size)
                                       + np.repeat([0, 0, _FORGET_BIAS],
                                                   size).astype(sloika_dtype)))
        self.p = th.shared(has_peep * init((3, size))/ np.sqrt(size))
        self.iW = th.shared(init((3 * size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((3 * size, size)) / np.sqrt(size + size))

    def params(self):
        params =  [self.iW, self.sW]
        if self.has_bias:
            params += [self.b]
        if self.has_peep:
            params += [self.p]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "LSTM-O"),
                           ('activation', self.fun.func_name),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias),
                           ('peep', self.has_peep)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW, (3, self.size, self.insize))),
                                         ('sW', _extract(self.sW, (3, self.size, self.size))),
                                         ('b', _extract(self.b, (3, self.size))),
                                         ('p', _extract(self.p, (3, self.size)))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape == (3, self.size)
            self.b = th.shared(values['b'].reshape(-1))
        if self.has_peep:
            assert values['p'].shape == (3, self.size)
            self.p = th.shared(values['p'])
        assert values['iW'].shape == (3, self.size, self.insize)
        self.iW = th.shared(values['iW'].reshape((3 * self.size, self.insize)))
        assert values['sW'].shape == (3, self.size, self.size)
        self.sW = th.shared(values['sW'].reshape((3 * self.size, self.size)))


    def step(self, in_vec, in_state):
        vW = T.tensordot(in_vec, self.iW, axes=(1, 1))
        outW = T.tensordot(in_state, self.sW, axes=(1, 1))
        sumW = vW + outW  + self.b
        sumW = sumW.reshape((-1, 3, self.size))

        #  Forget gate activation
        state = in_state * activation.sigmoid(sumW[:,2] + in_state * self.p[2])
        #  Update state with input
        state += self.fun(sumW[:,0] + in_state * self.p[0]) * activation.sigmoid(sumW[:,1] + in_state * self.p[1])
        return state


class Forget(RNN):
    """ Simple forget gate

    :params insize: Size of input to layer
    :params size: Layer size
    :params init: function to initialise tensors with
    :params has_bias: Whether layer has bias
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh):
        self.size = size
        self.insize = insize
        self.has_bias = has_bias
        self.fun = fun

        self.b = th.shared(has_bias * (init(2 * size)
                                       + np.repeat([_FORGET_BIAS, 0],
                                                   size).astype(sloika_dtype)))
        self.iW = th.shared(init((2 * size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((2 * size, size)) / np.sqrt(size + size))

    def params(self):
        params =  [self.iW, self.sW]
        if self.has_bias:
            params += [self.b]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "forget gate"),
                           ('activation',self.fun.func_name),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW, (2, self.size, self.insize))),
                                         ('sW', _extract(self.sW, (2, self.size, self.size))),
                                         ('b', _extract(self.b, (2, self.size)))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape == (2, self.size)
            self.b = th.shared(values['b'].reshape(-1))
        assert values['iW'].shape == (2, self.size, self.insize)
        self.iW = th.shared(values['iW'].reshape((2 * self.size, self.insize)))
        assert values['sW'].shape == (2, self.size, self.size)
        self.sW = th.shared(values['sW'].reshape((2 * self.size, self.size)))

    def step(self, in_vec, in_state):
        vI = T.tensordot(in_vec, self.iW, axes=(1,1))
        vS = T.tensordot(in_state, self.sW, axes=(1,1))
        vT = vI + vS + self.b
        vT = vT.reshape((-1, 2, self.size))

        forget = activation.sigmoid(vT[:,0])
        state = in_state * forget + (1.0 - forget) * self.fun(vT[:,1])
        return state


class Gru(RNN):
    """ Gated Recurrent Unit

    :params insize: Size of input to layer
    :params size: Layer size
    :params init: function to initialise tensors with
    :params has_bias: Whether layer has bias
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh):
        self.size = size
        self.insize = insize
        self.has_bias = has_bias
        self.fun = fun

        self.b = th.shared(has_bias * init(3 * size))
        self.iW = th.shared(init((3 * size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((2 * size, size)) / np.sqrt(size + size))
        self.sW2 = th.shared(init((size, size)) / np.sqrt(size + size))

    def params(self):
        params =  [self.iW, self.sW, self.sW2]
        if self.has_bias:
            params += [self.b]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "GRU"),
                           ('activation', self.fun.func_name),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW, (3, self.size, self.insize))),
                                         ('sW', _extract(self.sW, (2, self.size, self.size))),
                                         ('sW2', _extract(self.sW2)),
                                         ('b', _extract(self.b, (3, self.size)))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape == (3, self.size)
            self.b = th.shared(values['b'].reshape(-1))
        assert values['iW'].shape == (3, self.size, self.insize)
        self.iW = th.shared(values['iW'].reshape((3 * self.size, self.insize)))
        assert values['sW'].shape == (2, self.size, self.size)
        self.sW = th.shared(values['sW'].reshape((2 * self.size, self.size)))
        assert values['sW2'].shape == (self.size,  self.size)
        self.sW2 = th.shared(values['sW2'])

    def step(self, in_vec, in_state):
        vI = T.tensordot(in_vec, self.iW, axes=(1,1)) + self.b
        vS = T.tensordot(in_state, self.sW, axes=(1,1))
        vT = vI[:, :2 * self.size] + vS
        vT = vT.reshape((-1, 2, self.size))

        z = activation.sigmoid(vT[:,0])
        r = activation.sigmoid(vT[:,1])
        y = T.tensordot(r * in_state, self.sW2, axes=(1,1))
        hbar = self.fun(vI[:, 2 * self.size:] + y)
        state = z * in_state + (1 - z) * hbar
        return state


class Mut1(RNN):
    """ MUT1 from Jozefowicz
    http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf

    :params insize: Size of input to layer
    :params size: Layer size
    :params init: function to initialise tensors with
    :params has_bias: Whether layer has bias
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh):
        self.size = size
        self.insize = insize
        self.has_bias = has_bias
        self.fun = fun

        self.b = th.shared(has_bias * (init(2 * size)
                                       + np.repeat([_FORGET_BIAS, 0],
                                                   size).astype(sloika_dtype)))
        self.b2 = th.shared(has_bias * init(size))
        self.iW = th.shared(init((2 * size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((size, size)) / np.sqrt(size + size))
        self.sW2 = th.shared(init((size, size)) / np.sqrt(size + size))

    def params(self):
        params =  [self.iW, self.sW, self.sW2]
        if self.has_bias:
            params += [self.b, self.b2]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "MUT1"),
                           ('activation', self.fun.func_name),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW, (2, self.size, self.insize))),
                                         ('sW', _extract(self.sW)),
                                         ('sW2', _extract(self.sW2)),
                                         ('b', _extarct(self.b, (2, self.size))),
                                         ('b2', _extract(self.b2))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape == (2, self.size)
            self.b = th.shared(values['b'].reshape(-1))
            assert values['b2'].shape == (self.size)
            self.b2 = th.shared(values['b2'].reshape(-1))
        assert values['iW'].shape == (2, self.size, self.insize)
        self.iW = th.shared(values['iW'].reshape((2 * self.size, self.insize)))
        assert values['sW'].shape == (self.size, self.size)
        self.sW = th.shared(values['sW'])
        assert values['sW2'].shape == (self.size,  self.size)
        self.sW2 = th.shared(values['sW2'])

    def step(self, in_vec, in_state):
        vI = T.tensordot(in_vec, self.iW, axes=(1,1))
        vS = T.tensordot(in_state, self.sW, axes=(1,1))
        vT = vI + self.b
        vT = vT.reshape((-1, 2, self.size))

        z = activation.sigmoid(vT[:,0])
        r = activation.sigmoid(vT[:,1] + vS)
        y = T.tensordot(r * in_state, self.sW2, axes=(1,1))
        state = self.fun(y + self.fun(in_vec) + self.b2) * (1 - z) + z * in_state
        return state


class Reverse(Layer):
    """  Runs a recurrent layer in reverse time (backwards)
    """
    def __init__(self, layer):
       self.layer = layer

    def params(self):
        return self.layer.params()

    def json(self, params=False):
        return OrderedDict([('type', "reverse"),
                            ('sublayer', self.layer.json(params))])

    def set_params(self, values):
        return

    def run(self, inMat):
        return self.layer.run(inMat[::-1])[::-1]


class Parallel(Layer):
    """ Run multiple layers in parallel (all have same input and outputs are concatenated)
    """
    def __init__(self, layers):
        self.layers = layers

    def params(self):
        return reduce(lambda x, y: x + y.params(), self.layers, [])

    def json(self, params=False):
        return OrderedDict([('type', "parallel"),
                            ('sublayers', [layer.json(params) for layer in self.layers])])

    def set_params(self, values):
        return

    def run(self, inMat):
        return T.concatenate(map(lambda x: x.run(inMat), self.layers), axis=2)


class Serial(Layer):
    """ Run multiple layers serially: output of a layer is the input for the next layer
    """
    def __init__(self, layers):
        self.layers = layers

    def params(self):
        return reduce(lambda x, y: x + y.params(), self.layers, [])

    def json(self, params=False):
        return OrderedDict([('type', "serial"),
                            ('sublayers', [layer.json(params) for layer in self.layers])])

    def set_params(self, values):
        return

    def run(self, inMat):
        tmp = inMat
        for layer in self.layers:
            tmp = layer.run(tmp)
        return tmp


class Decode(RNN):
    """ Forward pass of a Viterbi decoder
    """
    def __init__(self, k):
        self.size = _NBASE ** k
        self.rstep = _NBASE ** (k - 1)
        self.rskip = _NBASE ** (k - 2)

    def params(self):
        return []

    def json(self, params=False):
        return OrderedDict([('type', "decode")])

    def set_params(self, values):
        return

    def step(self, in_vec, in_state):
        pscore = in_state[:,:self.size]
        # Stay
        score = pscore
        iscore = T.zeros_like(score)
        iscore += T.arange(0, stop=self.size)
        # Step
        pscore = pscore.reshape((-1, _NSTEP, self.rstep))
        score2 = T.repeat(T.max(pscore, axis=1), _NSTEP)
        iscore2 = T.repeat(self.rstep * T.argmax(pscore, axis=1) + T.arange(0, stop=self.rstep, dtype=sloika_dtype), _NSTEP)
        iscore2 = iscore2.reshape((-1, self.size))
        score2 = score2.reshape((-1, self.size))
        iscore = T.switch(T.gt(score, score2), iscore, iscore2)
        score = T.maximum(score, score2)
        # Skip
        pscore = pscore.reshape((-1, _NSKIP, self.rskip))
        score2 = T.repeat(T.max(pscore, axis=1), _NSKIP)
        iscore2 = T.repeat(self.rstep * T.argmax(pscore, axis=1) + T.arange(0, stop=self.rskip), _NSKIP)
        iscore2 = iscore2.reshape((-1, self.size))
        score2 = score2.reshape((-1, self.size))
        iscore = T.switch(T.gt(score, score2), iscore, iscore2)
        score = T.maximum(score, score2)

        score += T.log(T.nnet.softmax(in_vec))
        return T.concatenate((iscore, score), axis=1)

    def run(self, inMat):
        nbatch = T.shape(inMat)[1]
        out, _ = th.scan(self.step, sequences=inMat,
                         outputs_info=T.zeros((nbatch, 2 * self.size)))
        return out[:,:,self.size]

def birnn(layer1, layer2):
    """  Creates a bidirectional RNN from two RNNs
    """
    return Parallel([layer1, Reverse(layer2)])
