import abc
import cPickle
import json
import numpy as np
import tempfile
import unittest

import theano as th
import theano.tensor as T

from sloika import activation, sloika_dtype
import sloika.layers as nn

class ANNTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print '* Layers'
        np.random.seed(0xdeadbeef)
        self._NSTEP = 100
        self._NFEATURES = 3
        self._SIZE = 64
        self._NBATCH = 2

        self.W = np.random.normal(size=(self._SIZE, self._NFEATURES)).astype(sloika_dtype)
        self.b = np.random.normal(size=self._SIZE).astype(sloika_dtype)
        self.x = np.random.normal(size=(self._NSTEP, self._NBATCH, self._NFEATURES)).astype(sloika_dtype)
        self.res = self.x.dot(self.W.transpose()) + self.b

    def test_000_single_layer_linear(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True,
                                 fun=activation.linear)
        network.set_params({ 'W': self.W, 'b': self.b})
        f = network.compile()
        np.testing.assert_almost_equal(f(self.x), self.res, decimal=5)

    def test_001_single_layer_tanh(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        network.set_params({ 'W': self.W, 'b': self.b})
        f = network.compile()
        np.testing.assert_almost_equal(f(self.x), np.tanh(self.res), decimal=5)

    def test_002_parallel_layers(self):
        l1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        l1.set_params({ 'W': self.W, 'b': self.b})
        l2 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        l2.set_params({ 'W': self.W, 'b': self.b})
        network = nn.Parallel([l1, l2])
        f = network.compile()

        res = f(self.x)
        np.testing.assert_almost_equal(res[:,:,:self._SIZE], res[:,:,self._SIZE:])

    def test_003_simple_serial(self):
        W2 = np.random.normal(size=(self._SIZE, self._SIZE)).astype(sloika_dtype)
        res = self.res.dot(W2.transpose())

        l1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True,
                            fun=activation.linear)
        l1.set_params({ 'W': self.W, 'b': self.b})
        l2 = nn.FeedForward(self._SIZE, self._SIZE, fun=activation.linear)
        l2.set_params({ 'W': W2})
        network = nn.Serial([l1, l2])
        f = network.compile()

        np.testing.assert_almost_equal(f(self.x), res, decimal=4)

    def test_004_reverse(self):
        network1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        network1.set_params({ 'W': self.W, 'b': self.b})
        f1 = network1.compile()
        res1 = f1(self.x)
        network2 = nn.Reverse(network1)
        f2 = network2.compile()
        res2 = f2(self.x)

        np.testing.assert_almost_equal(res1, res2)

    def test_005_poormans_birnn(self):
        layer1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        layer1.set_params({ 'W': self.W, 'b': self.b})
        layer2 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        layer2.set_params({ 'W': self.W, 'b': self.b})
        network = nn.birnn(layer1, layer2)
        f = network.compile()

        res = f(self.x)
        np.testing.assert_almost_equal(res[:,:,:self._SIZE], res[:,:,self._SIZE:])

    def test_006_softmax(self):
        network = nn.Softmax(self._NFEATURES, self._SIZE, has_bias=True)
        network.set_params({ 'W': self.W, 'b': self.b})
        f = network.compile()

        res = f(self.x)
        res_sum = res.sum(axis=2)
        self.assertTrue(np.allclose(res_sum, 1.0))

    def test_007_rnn_no_state(self):
        sW = np.zeros((self._SIZE, self._SIZE), dtype=sloika_dtype)
        network = nn.Recurrent(self._NFEATURES, self._SIZE, has_bias=True,
                               fun=activation.linear)
        network.set_params({ 'iW': self.W, 'sW': sW, 'b': self.b})
        f = network.compile()

        res = f(self.x)
        np.testing.assert_almost_equal(res, self.res, decimal=5)

    def test_008_rnn_no_input(self):
        iW = np.zeros((self._SIZE, self._NFEATURES), dtype=sloika_dtype)
        sW = np.random.normal(size=(self._SIZE, self._SIZE)).astype(sloika_dtype)
        network = nn.Recurrent(self._NFEATURES, self._SIZE)
        network.set_params({ 'iW': iW, 'sW': sW})
        f = network.compile()

        res = f(self.x)
        np.testing.assert_almost_equal(res, 0.0)

    def test_009_rnn_no_input_with_bias(self):
        iW = np.zeros((self._SIZE, self._NFEATURES), dtype=sloika_dtype)
        sW = np.random.normal(size=(self._SIZE, self._SIZE)).astype(sloika_dtype)
        network = nn.Recurrent(self._NFEATURES, self._SIZE, has_bias=True,
                               fun=activation.linear)
        network.set_params({ 'iW': iW, 'sW': sW, 'b': self.b})
        f = network.compile()

        res = f(self.x)
        res2 = np.zeros((self._NBATCH,self._SIZE))
        for i in xrange(self._NSTEP):
            res2 = res2.dot(sW.transpose()) + self.b
            np.testing.assert_almost_equal(res[i], res2)

    def test_010_birnn_no_input_with_bias(self):
        iW = np.zeros((self._SIZE, self._NFEATURES), dtype=sloika_dtype)
        sW = np.random.normal(size=(self._SIZE, self._SIZE)).astype(sloika_dtype)
        layer1 = nn.Recurrent(self._NFEATURES, self._SIZE, has_bias=True,
                              fun=activation.linear)
        layer1.set_params({ 'iW': iW, 'sW': sW, 'b': self.b})
        layer2 = nn.Recurrent(self._NFEATURES, self._SIZE, has_bias=True,
                              fun=activation.linear)
        layer2.set_params({ 'iW': iW, 'sW': sW, 'b': self.b})
        network = nn.birnn(layer1, layer2)

        f = network.compile()

        res = f(self.x)
        np.testing.assert_almost_equal(res[:,:,:self._SIZE], res[::-1,:,self._SIZE:])

    def test_012_simple_derivative(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE,
                                 fun=activation.linear)
        network.set_params({'W': self.W})
        params = network.params()
        x = T.tensor3()
        loss = T.sum(network.run(x))
        grad = th.gradient.jacobian(loss, params)
        f = th.function([x], grad)

        theano_grad = f(self.x)[0]
        analytic_grad = np.sum(self.x, axis=(0, 1))
        self.assertEqual(theano_grad.shape, (self._SIZE, self._NFEATURES))
        for i in xrange(self._NFEATURES):
            np.testing.assert_almost_equal(theano_grad[i], analytic_grad, decimal=3)

    def test_013_simple_derivative_with_bias(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True,
                                 fun=activation.linear)
        network.set_params({'W': self.W, 'b': self.b})
        params = network.params()
        x = T.tensor3()
        loss = T.sum(network.run(x))
        grad = th.gradient.jacobian(loss, params)
        f = th.function([x], grad)

        theano_grad = f(self.x)
        W_grad = np.sum(self.x, axis=(0, 1))
        self.assertEqual(theano_grad[0].shape, (self._SIZE, self._NFEATURES))
        self.assertEqual(theano_grad[1].shape[0], self._SIZE)
        for i in xrange(self._NFEATURES):
            np.testing.assert_almost_equal(theano_grad[0][i], W_grad, decimal=3)
            np.testing.assert_almost_equal(theano_grad[1][i], self._NBATCH * self._NSTEP)

    def test_014_complex_derivative(self):
        iW = np.random.normal(size=(4, self._SIZE, self._NFEATURES)).astype(sloika_dtype)
        sW = np.random.normal(size=(4, self._SIZE, self._SIZE)).astype(sloika_dtype)
        b = np.random.normal(size=(4, self._SIZE)).astype(sloika_dtype)
        p = np.random.normal(size=(3, self._SIZE)).astype(sloika_dtype)
        network = nn.Lstm(self._NFEATURES, self._SIZE, has_bias=True, has_peep=True)
        network.set_params({'iW': iW, 'sW': sW, 'b': b, 'p': p})

        x = T.tensor3()
        loss = T.sum(network.run(x))
        grad = th.gradient.jacobian(loss, network.params())
        f = th.function([x], grad)

        theano_grad = f(self.x)[0]

    def test_015_save_then_load(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE,
                                 fun=activation.linear)
        network.set_params({'W': self.W})
        params = network.params()
        x = T.tensor3()
        loss = T.sum(network.run(x))
        grad = th.grad(loss, params)
        f = th.function([x], grad)
        with tempfile.TemporaryFile() as tf:
            cPickle.dump(f, tf)
            tf.seek(0)
            f2 = cPickle.load(tf)

        theano_grad = f(self.x)[0]
        theano_grad2 = f2(self.x)[0]
        self.assertEqual(theano_grad.shape, (self._SIZE, self._NFEATURES))
        self.assertEqual(theano_grad2.shape, (self._SIZE, self._NFEATURES))
        np.testing.assert_almost_equal(theano_grad, theano_grad2, decimal=3)

    def test_016_window(self):
        _WINLEN = 3
        network = nn.Window(_WINLEN)
        f = network.compile()
        res = f(self.x)
        #  Window is now 'SAME' not 'VALID'. Trim
        wh = _WINLEN // 2
        res = res[wh : -wh]
        for j in xrange(self._NBATCH):
            for i in xrange(_WINLEN - 1):
                try:
                    np.testing.assert_almost_equal(res[: ,j ,i * _WINLEN : (i + 1) * _WINLEN], self.x[i : 1 + i - _WINLEN, j])
                except:
                    print "Window max: {}".format(np.amax(np.fabs(res[:,:,i * _WINLEN : (i + 1) * _WINLEN] - self.x[ i : 1 + i - _WINLEN])))
                    raise
            np.testing.assert_almost_equal(res[: ,j ,_WINLEN  * (_WINLEN - 1) :], self.x[_WINLEN - 1 :, j])
            #  Test first and last rows explicitly
            np.testing.assert_almost_equal(self.x[:_WINLEN, j].ravel(), res[0, j].transpose().ravel())
            np.testing.assert_almost_equal(self.x[-_WINLEN:, j].ravel(), res[-1, j].transpose().ravel())

    def test_017_decode_simple(self):
        _KMERLEN = 3
        network = nn.Decode(_KMERLEN)
        f = network.compile()
        res = f(self.res)

    def test_018_studentise(self):
        network = nn.Studentise()
        f = network.compile()
        res = f(self.x)

        np.testing.assert_almost_equal(np.mean(res, axis=(0, 1)), 0.0)
        np.testing.assert_almost_equal(np.std(res, axis=(0, 1)), 1.0, decimal=4)

    def test_019_identity(self):
        network = nn.Identity()
        f = network.compile()
        res = f(self.res)

        np.testing.assert_almost_equal(res, self.res)

class LayerTest(object):
    """Mixin abstract class for testing basic layer functionality"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def run_inputs(self):
        """List of input arrays for testing layer.run"""
        return []

    @abc.abstractproperty
    def learned_parameter_names(self):
        """List of names of learned parameters"""
        return []

    @property
    def init(self):
        """Initialisation function for numpy arrays to be used in testing"""
        return lambda dims: np.zeros(dims, dtype=sloika_dtype)

    @abc.abstractmethod
    def setUp(self):
        """Create the layer as self.layer"""
        return

    def test_000_run(self):
        outs = [self.layer.run(In) for In in self.run_inputs]

    def test_001_compile(self):
        out = self.layer.compile()

    def test_002_json_dumps(self):
        js = json.dumps(self.layer.json())

    def test_003_json_decodes(self):
        props = json.JSONDecoder().decode(json.dumps(self.layer.json()))

    def test_004_get_set_params(self):
        p0 = self.layer.json(params=True)["params"]
        p0 = {p: np.array(v) for (p, v) in p0.items()}
        for (p, v) in p0.items():
            p0[p] = v + np.random.uniform(size=v.shape)
        self.layer.set_params(p0)
        p1 = self.layer.json(params=True)["params"]
        p1 = {p: np.array(v) for (p, v) in p1.items()}
        self.assertTrue(all([np.allclose(p0[k], p1[k]) for k in self.learned_parameter_names]))

class RNNTest(LayerTest):
    """Mixin abstract class for testing basic RNN functionality"""
    pass

class RecurrentTest(RNNTest, unittest.TestCase):
    def setUp(self):
        self.layer = nn.Recurrent(12, 64)

    @property
    def run_inputs(self):
        return [np.random.uniform(size=(10, 20, 12)),]

    @property
    def learned_parameter_names(self):
        return ['iW', 'sW',]

class RecurrentBiasedTest(RNNTest, unittest.TestCase):
    def setUp(self):
        self.layer = nn.Recurrent(12, 64, has_bias=True)

    @property
    def run_inputs(self):
        return [np.random.uniform(size=(10, 20, 12)),]

    @property
    def learned_parameter_names(self):
        return ['iW', 'sW', 'b',]

class LstmTest(RNNTest, unittest.TestCase):
    def setUp(self):
        self.layer = nn.Lstm(12, 64)

    @property
    def run_inputs(self):
        return [np.random.uniform(size=(10, 20, 12)),]

    @property
    def learned_parameter_names(self):
        return ['iW', 'sW',]

class LstmCIFGTest(RNNTest, unittest.TestCase):
    def setUp(self):
        self.layer = nn.LstmCIFG(12, 64)

    @property
    def run_inputs(self):
        return [np.random.uniform(size=(10, 20, 12)),]

    @property
    def learned_parameter_names(self):
        return ['iW', 'sW',]

class LstmOTest(RNNTest, unittest.TestCase):
    def setUp(self):
        self.layer = nn.LstmO(12, 64)

    @property
    def run_inputs(self):
        return [np.random.uniform(size=(10, 20, 12)),]

    @property
    def learned_parameter_names(self):
        return ['iW', 'sW',]

class Mut1Test(RNNTest, unittest.TestCase):
    def setUp(self):
        self.layer = nn.Mut1(12, 64)

    @property
    def run_inputs(self):
        return [np.zeros((10, 20, 12)),
                np.random.uniform(size=(10, 20, 12)),]

    @property
    def learned_parameter_names(self):
        return ['W_xz', 'W_xr', 'W_hr', 'W_hh',]

class Mut2Test(RNNTest, unittest.TestCase):
    def setUp(self):
        self.layer = nn.Mut2(12, 64)

    @property
    def run_inputs(self):
        return [np.zeros((10, 20, 12)),
                np.random.uniform(size=(10, 20, 12)),]

    @property
    def learned_parameter_names(self):
        return ['W_xz', 'W_hz', 'W_hr', 'W_hh', 'W_xh',]

class Mut3Test(RNNTest, unittest.TestCase):
    def setUp(self):
        self.layer = nn.Mut3(12, 64)

    @property
    def run_inputs(self):
        return [np.zeros((10, 20, 12)),
                np.random.uniform(size=(10, 20, 12)),]

    @property
    def learned_parameter_names(self):
        return ['W_xz', 'W_hz', 'W_xr', 'W_hr', 'W_hh', 'W_xh',]

class GruTest(RNNTest, unittest.TestCase):
    def setUp(self):
        self.layer = nn.Gru(12, 64)

    @property
    def run_inputs(self):
        return [np.random.uniform(size=(10, 20, 12)),]

    @property
    def learned_parameter_names(self):
        return ['iW', 'sW', 'sW2',]

class ScrnTest(RNNTest, unittest.TestCase):
    def setUp(self):
        self.layer = nn.SCRN(12, 48, 16)

    @property
    def run_inputs(self):
        return [np.random.uniform(size=(10, 20, 12)),]

    @property
    def learned_parameter_names(self):
        return ['isW', 'sfW', 'ifW', 'ffW',]
