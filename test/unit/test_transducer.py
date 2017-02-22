from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
from past.utils import old_div
import numpy as np
import sys
from sloika.transducer import align, alignment_to_call
import unittest

_NEGLARGE = -3000.0
_PRINT = False


class TransducerTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print('* Decode 2D transducer')
        np.random.seed(0xdeadbeef)
        self.n = 7
        self.gap = -5.0
        self.scale = 0.001

    def _fill_seq(self, seq):
        post = np.zeros((len(seq), 5))
        post.fill(_NEGLARGE)
        for i, s in enumerate(seq):
            post[i, s] = 0.0
        post += np.random.uniform(size=post.shape) * self.scale
        return post

    def _compare_seqs(self, seq1, seq2):
        post1 = self._fill_seq(seq1)
        post2 = self._fill_seq(seq2)
        score, alignment = align(post1, post2, old_div(self.gap, 2.0), self.gap, old_div(self.gap, 2.0))
        path = alignment_to_call(post1, post2, alignment)
        if _PRINT:
            print('* ', sys._getframe(1).f_code.co_name)
            print('  input sequences', seq1, seq2)
            print('  result', score, alignment, path)
        return score, alignment, path

    def test_001_align_A(self):
        score, alignment, path = self._compare_seqs([3] * self.n, [0] * self.n)
        self.assertTrue(np.array_equiv(alignment, 0))
        self.assertTrue(np.array_equiv(path, 0))

    def test_002_align_C(self):
        score, alignment, path = self._compare_seqs([2] * self.n, [1] * self.n)
        self.assertTrue(np.array_equiv(alignment, 0))
        self.assertTrue(np.array_equiv(path, 1))

    def test_003_align_G(self):
        score, alignment, path = self._compare_seqs([1] * self.n, [2] * self.n)
        self.assertTrue(np.array_equiv(alignment, 0))
        self.assertTrue(np.array_equiv(path, 2))

    def test_004_align_T(self):
        score, alignment, path = self._compare_seqs([0] * self.n, [3] * self.n)
        self.assertTrue(np.array_equiv(alignment, 0))
        self.assertTrue(np.array_equiv(path, 3))

    def test_005_align_palindrome(self):
        seq1 = [0, 1, 2, 3, 2, 1, 0]
        seq2 = [3, 2, 1, 0, 1, 2, 3]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, 0))
        self.assertTrue(np.array_equiv(path, seq2))

    def test_006_align_repeat(self):
        seq1 = [0, 1, 2, 3, 0, 1, 2]
        seq2 = [1, 2, 3, 0, 1, 2, 3]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, 0))
        self.assertTrue(np.array_equiv(path, seq2))

    def test_007_align_central_stay(self):
        seq1 = [0, 1, 2, 4, 0, 1, 2]
        seq2 = [1, 2, 3, 4, 1, 2, 3]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, 0))
        self.assertTrue(np.array_equiv(path, seq2))

    def test_008_align_mathcing_offcentre_stay(self):
        seq1 = [0, 1, 4, 3, 0, 1, 2]
        seq2 = [1, 2, 3, 0, 4, 2, 3]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, 0))
        self.assertTrue(np.array_equiv(path, seq2))

    def test_009_align_inconsistent_offcentre_stay(self):
        seq1 = [0, 3, 4, 3, 0, 1, 2]
        seq2 = [1, 2, 4, 0, 1, 0, 3]
        ans = [0, 0, 3, 2, 0, 1, 4, 0, 0]
        call = [1, 2, 4, 3, 0, 4, 1, 0, 3]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, ans))
        self.assertTrue(np.array_equiv(path, call))

    @unittest.skip
    def test_010_align_error(self):
        seq1 = [0, 1, 2, 3, 3, 1, 2]
        seq2 = [1, 2, 3, 0, 1, 2, 3]
        ans = [0, 0, 4, 2, 0, 0, 0, 0]
        call = [1, 2, 3, 0, 0, 1, 2, 3]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, ans))
        self.assertTrue(np.array_equiv(path, call))

    def test_011_align_alldiff(self):
        seq1 = [0] * self.n
        seq2 = [1] * self.n
        ans = [4] * self.n + [2] * self.n
        call = [1] * self.n + [3] * self.n
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, ans))
        self.assertTrue(np.array_equiv(path, call))

    def test_012_align_excess1(self):
        seq1 = [1, 0, 1, 3, 2, 0, 1, 0, 3]
        seq2 = [3, 2, 3, 1, 0, 2]
        ans = [2] + [0] * 6 + [2, 2]
        call = [0, 3, 2, 3, 1, 0, 2, 3, 2]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, ans))
        self.assertTrue(np.array_equiv(path, call))

    def test_013_align_excess2(self):
        seq1 = [3, 1, 3, 1, 0, 2]
        seq2 = [1, 0, 1, 3, 2, 0, 2, 0, 3]
        ans = [4, 4] + [0] * 6 + [4]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, ans))
        self.assertTrue(np.array_equiv(path, seq2))

    def test_014_align_overlap1(self):
        seq1 = [0, 3, 1, 3, 1, 0, 2]
        seq2 = [1, 0, 1, 3, 2, 0, 2]
        ans = [4, 4] + [0] * 5 + [2, 2]
        call = [1, 0, 1, 3, 2, 0, 2, 0, 3]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, ans))
        self.assertTrue(np.array_equiv(path, call))

    def test_015_align_overlap2(self):
        seq1 = [1, 3, 1, 0, 2, 3, 2]
        seq2 = [1, 3, 2, 0, 2, 0, 3]
        ans = [2, 2] + [0] * 5 + [4, 4]
        call = [1, 0, 1, 3, 2, 0, 2, 0, 3]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, ans))
        self.assertTrue(np.array_equiv(path, call))

    @unittest.skip
    def test_016_align_errorstay1(self):
        seq1 = [0, 1, 2, 3, 3, 1, 2]
        seq2 = [1, 2, 4, 0, 1, 2, 3]
        ans = [0, 0, 3, 2, 0, 0, 0, 0]
        call = [1, 2, 4, 0, 0, 1, 2, 3]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(alignment, ans))
        self.assertTrue(np.array_equiv(path, call))

    def test_017_align_errorstay2(self):
        seq1 = [0, 1, 2, 3, 4, 1, 2]
        seq2 = [1, 2, 3, 0, 1, 2, 3]
        ans = [0, 0, 1, 4, 0, 0, 0, 0]
        call = [1, 2, 4, 3, 0, 1, 2, 3]
        score, alignment, path = self._compare_seqs(seq1, seq2)
        self.assertTrue(np.array_equiv(path, call))
