import numpy as np
import unittest

from sloika import decode

class TestDecode(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print '* Decoding'
        self.post = [[0.144983872, 0.0353539565, 0.460170397, 0.0003722599, 0.3591195148],
                     [0.100967586, 0.0357787755, 0.003763944, 0.0135964994, 0.8458931946],
                     [0.225580112, 0.0053868825, 0.127545423, 0.0438386941, 0.5976488894],
                     [0.034071887, 0.0124396516, 0.390811281, 0.0058303676, 0.5568468128],
                     [0.070028528, 0.3403599935, 0.157938013, 0.3416912224, 0.0899822435],
                     [0.010880335, 0.8579484836, 0.112103479, 0.0185191681, 0.0005485341],
                     [0.009025176, 0.8074192531, 0.039663213, 0.0830854627, 0.0608068949],
                     [0.141001418, 0.3820869847, 0.179637615, 0.2329239763, 0.0643500054],
                     [0.226134609, 0.2082560019, 0.481295410, 0.0826148125, 0.0016991672],
                     [0.048037662, 0.0004689463, 0.354844142, 0.0071289458, 0.5895203039]]
        self.post = np.array(self.post)
        self.labels = np.array([2, 4, 4, 4, 3, 1, 1, 1, 2, 4])
        self.bases = np.array([2, 3, 1, 1, 1, 2])
        self.score = -4.4275354890527474
        self.score_full = -5.0702616325672301
        self.score_viterbi = -5.70653594347

    def test_001_argmax(self):
        bases = decode.argmax(self.post)
        self.assertEqual(len(bases), len(self.bases))
        self.assertTrue(np.array_equiv(bases, self.bases))

    def test_002_score(self):
        score = decode.score(self.post, self.bases)
        self.assertAlmostEqual(score, self.score)

    def test_003_score_full_length(self):
        score = decode.score(self.post, self.bases, full=True)
        self.assertAlmostEqual(score, self.score_full)

    def test_004_score_ordering(self):
        bases = decode.argmax(self.post)
        score1 = decode.score(self.post, bases)
        score2 = decode.score(self.post, bases, full=True)
        vpath = np.argmax(self.post, axis=1)
        vscore = np.sum(np.log([p[vp] for p, vp in zip(self.post, vpath)]))

        self.assertGreaterEqual(score1, score2)
        self.assertGreaterEqual(score2, vscore)

    def test_005_transposed_score(self):
        score = decode.forwards_transpose(self.post, self.bases)
        self.assertAlmostEqual(score, self.score_full)

    def test_006_forward_equals_backwards(self):
        bases = decode.argmax(self.post)
        scoreF = decode.forwards_transpose(self.post, bases)
        scoreB = decode.backwards_transpose(self.post, bases)
        self.assertAlmostEqual(scoreF, scoreB)
