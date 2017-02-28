from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import unittest
import os
import shutil
import h5py
import shutil
import tempfile
import difflib

import numpy as np

from distutils import dir_util
from nose_parameterized import parameterized


from util import run_cmd, is_close, maybe_create_dir, zeroth_line_starts_with, last_line_starts_with


class AcceptanceTest(unittest.TestCase):
    known_commands = ["identity", "remap"]

    @classmethod
    def setUpClass(self):
        self.test_directory = os.path.splitext(__file__)[0]
        self.test_name = os.path.basename(self.test_directory)
        self.script = os.path.join(os.environ["BIN_DIR"], "chunkify.py")

        self.work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], self.test_name)
        maybe_create_dir(self.work_dir)

        self.data_dir = os.path.join(os.environ["DATA_DIR"], self.test_name)

    def assertClose(self, a, b):
        if not is_close(a, b, 1e-5):
            msg = '{} is not close {}'.format(a, b)
            self.assertTrue(is_close(a, b, 1e-5), msg)

    def test_commands(self):
        cmd = [self.script]
        run_cmd(self, cmd).return_code(0).stdout(zeroth_line_starts_with(u"Available commands:"))

    @parameterized.expand(known_commands)
    def test_usage(self, command_name):
        cmd = [self.script, command_name]
        run_cmd(self, cmd).return_code(2).stderr(zeroth_line_starts_with(u"usage:"))

    def test_unsupported_command(self):
        cmd = [self.script, "hehe"]
        run_cmd(self, cmd).return_code(1).stdout(zeroth_line_starts_with(u"Unsupported command 'hehe'"))

    @parameterized.expand([
        [[], (182, 500, 4), -2.8844583, 14.225174, -0.254353493452],
        [["--normalisation", "per-read"], (182, 500, 4), -2.8844583, 14.225174, -0.254353493452],
        [["--normalisation", "per-chunk"], (182, 500, 4), -4.1303601265, 12.2556829453, -0.249717712402],
    ])
    def test_chunkify_with_identity_with_normalisation(self, options, chunks_shape, min_value, max_value, median_value):
        strand_input_list = os.path.join(self.data_dir, "identity", "na12878_train.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "identity", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as fh:
            output_file_name = fh.name

        cmd = [self.script, "identity", "--chunk_len", "500", "--kmer_len", "5",
               "--section", "template", "--input_strand_list", strand_input_list,
               reads_dir, output_file_name] + options

        run_cmd(self, cmd).return_code(1)

        run_cmd(self, cmd + ['--overwrite']).return_code(0)

        with h5py.File(output_file_name, 'r') as fh:
            top_level_items = []
            for item in fh:
                top_level_items.append(item)
            top_level_items.sort()
            self.assertEqual(top_level_items, [u'bad', u'chunks', u'labels', u'weights'])

            self.assertEqual(fh['chunks'].shape, chunks_shape)
            chunks = fh['chunks'][:]
            self.assertClose(chunks.min(), min_value)
            self.assertClose(chunks.max(), max_value)
            self.assertClose(np.median(chunks), median_value)

        os.remove(output_file_name)

    @parameterized.expand([
        [[], (33, 500, 4), -2.70142698288, 12.7569065094, -0.238316237926],
        [["--normalisation", "per-read"], (33, 500, 4), -2.70142698288, 12.7569065094, -0.238316237926],
        [["--normalisation", "per-chunk"], (33, 500, 4), -2.85386562347, 11.4694499969, -0.237461671233],
    ])
    def test_chunkify_with_remap_with_normalisation(self, options, chunks_shape, min_value, max_value, median_value):
        strand_input_list = os.path.join(self.data_dir, "remap", "strand_output_list.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "remap", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        model_file = os.path.join(self.data_dir, "remap", "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        reference_file = os.path.join(self.data_dir, "remap", "reference.fa")
        self.assertTrue(os.path.exists(reference_file))

        with tempfile.NamedTemporaryFile(prefix="strand_output_list", suffix=".txt", delete=False) as fh:
            strand_output_list = fh.name

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as fh:
            output_file_name = fh.name

        cmd = [self.script, "remap", "--trim", "200", "200", "--chunk_len", "500", "--kmer_len", "5",
               "--section", "template", "--input_strand_list", strand_input_list,
               "--output_strand_list",
               strand_output_list, reads_dir, output_file_name, model_file, reference_file] + options

        run_cmd(self, cmd).return_code(1)

        os.remove(output_file_name)

        run_cmd(self, cmd).return_code(2)

        run_cmd(self, cmd + ['--overwrite']).return_code(0)

        with h5py.File(output_file_name, 'r') as fh:
            top_level_items = []
            for item in fh:
                top_level_items.append(item)
            top_level_items.sort()
            self.assertEqual(top_level_items, [u'bad', u'chunks', u'labels', u'weights'])

            self.assertEqual(fh['chunks'].shape, chunks_shape)
            chunks = fh['chunks'][:]
            self.assertClose(chunks.min(), min_value)
            self.assertClose(chunks.max(), max_value)
            self.assertClose(np.median(chunks), median_value)

        os.remove(output_file_name)
        os.remove(strand_output_list)

    def test_chunkify_with_remap_no_results_due_to_missing_reference(self):
        strand_input_list = os.path.join(self.data_dir, "remap", "strand_output_list.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "identity", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        model_file = os.path.join(self.data_dir, "remap", "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        reference_file = os.path.join(self.data_dir, "remap", "reference.fa")
        self.assertTrue(os.path.exists(reference_file))

        with tempfile.NamedTemporaryFile(prefix="strand_output_list", suffix=".txt", delete=False) as fh:
            strand_output_list = fh.name

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as fh:
            output_file_name = fh.name

        cmd = [self.script, "remap", "--trim", "200", "200", "--chunk_len", "500", "--kmer_len", "5",
               "--section", "template", "--input_strand_list", strand_input_list,
               "--output_strand_list", strand_output_list,
               reads_dir, output_file_name, model_file, reference_file]

        os.remove(output_file_name)
        os.remove(strand_output_list)

        run_cmd(self, cmd).return_code(1).stderr(last_line_starts_with(u"no chunks were produced"))

        self.assertTrue(not os.path.exists(output_file_name))
        self.assertTrue(not os.path.exists(strand_output_list))

    @parameterized.expand([
        [500, 545, 25, 20, 0],
        [501, 545, 25, 20, 1],
        [500, 546, 25, 20, 1],
        [500, 545, 26, 20, 1],
        [500, 545, 25, 21, 1],
    ])
    def test_chunkify_with_remap_no_results_due_to_length(self, chunk_len, min_length, trim_left, trim_right, return_code):
        strand_input_list = os.path.join(self.data_dir, "remap2", "strand_input_list.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "remap2", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        model_file = os.path.join(self.data_dir, "remap2", "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        reference_file = os.path.join(self.data_dir, "remap2", "reference.fa")
        self.assertTrue(os.path.exists(reference_file))

        with tempfile.NamedTemporaryFile(prefix="strand_output_list", suffix=".txt", delete=False) as fh:
            strand_output_list = fh.name

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as fh:
            output_file_name = fh.name

        cmd = [self.script, "remap", "--trim", str(trim_left), str(trim_right), "--chunk_len", str(chunk_len), "--kmer_len", "5",
               "--section", "template", "--input_strand_list", strand_input_list,
               "--output_strand_list", strand_output_list, "--min_length", str(min_length),
               reads_dir, output_file_name, model_file, reference_file]

        os.remove(output_file_name)
        os.remove(strand_output_list)

        expectation = run_cmd(self, cmd).return_code(return_code)

        if return_code != 0:
            expectation.stderr(last_line_starts_with(u"no chunks were produced"))

            self.assertTrue(not os.path.exists(output_file_name))
            self.assertTrue(not os.path.exists(strand_output_list))

    @parameterized.expand([
        [300, 360, 40, 20, 0],
        [301, 360, 40, 20, 1],
        [300, 361, 40, 20, 1],
        [300, 360, 41, 20, 1],
        [300, 360, 40, 21, 1],
    ])
    def test_chunkify_with_identity_no_results_due_to_length(self, chunk_len, min_length, trim_left, trim_right, return_code):
        strand_input_list = os.path.join(self.data_dir, "remap2", "strand_input_list.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "remap2", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as fh:
            output_file_name = fh.name

        cmd = [self.script, "identity", "--trim", str(trim_left), str(trim_right), "--chunk_len", str(chunk_len), "--kmer_len", "5",
               "--section", "template", "--input_strand_list", strand_input_list, "--min_length", str(min_length),
               reads_dir, output_file_name]

        os.remove(output_file_name)

        expectation = run_cmd(self, cmd).return_code(return_code)

        if return_code != 0:
            expectation.stderr(last_line_starts_with(u"no chunks were produced"))

            self.assertTrue(not os.path.exists(output_file_name))
