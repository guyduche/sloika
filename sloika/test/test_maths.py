import unittest
import numpy as np
from sloika import maths

class MathsTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print '* Maths routines'
        np.random.seed(0xdeadbeef)

    def test_001_studentise(self):
        sh = (7, 4)
        x = np.random.normal(size=sh)
        x2 = maths.studentise(x)
        self.assertTrue(x2.shape == sh)
        self.assertAlmostEqual(np.mean(x2), 0.0)
        self.assertAlmostEqual(np.std(x2), 1.0)

    def test_002_studentise_over_axis0(self):
        sh = (7, 4)
        x = np.random.normal(size=sh)
        x2 = maths.studentise(x, axis=0)
        self.assertTrue(x2.shape == sh)
        self.assertTrue(np.allclose(np.mean(x2, axis=0), 0.0))
        self.assertTrue(np.allclose(np.std(x2, axis=0), 1.0))

    def test_003_studentise_over_axis1(self):
        sh = (7, 4)
        x = np.random.normal(size=sh)
        x2 = maths.studentise(x, axis=1)
        self.assertTrue(x2.shape == sh)
        self.assertTrue(np.allclose(np.mean(x2, axis=1), 0.0))
        self.assertTrue(np.allclose(np.std(x2, axis=1), 1.0))

