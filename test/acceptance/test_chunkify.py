from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

from distutils import dir_util
import h5py
from nose_parameterized import parameterized
import numpy as np
import os
import unittest

import util
from sloika.util import is_close


class AcceptanceTest(unittest.TestCase):
    known_commands = ["identity", "remap", "raw_identity", "raw_remap"]

    @classmethod
    def setUpClass(self):
        testset_directory = os.path.splitext(__file__)[0]
        testset_name = os.path.basename(testset_directory)

        self.data_dir = os.path.join(os.environ["DATA_DIR"], testset_name)

        self.testset_work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], testset_name)

        self.script = os.path.join(os.environ["BIN_DIR"], "chunkify.py")

    def work_dir(self, test_name):
        directory = os.path.join(self.testset_work_dir, test_name)
        util.maybe_create_dir(directory)
        return directory

    def assertClose(self, a, b):
        if not is_close(a, b, 1e-5):
            msg = '{} is not close {}'.format(a, b)
            self.assertTrue(is_close(a, b, 1e-5), msg)

    def test_commands(self):
        cmd = [self.script, "--help"]
        util.run_cmd(self, cmd).expect_exit_code(0).expect_stdout(util.zeroth_line_starts_with(u"usage:"))

    @parameterized.expand(known_commands)
    def test_usage(self, command_name):
        cmd = [self.script, command_name, "--help"]
        util.run_cmd(self, cmd).expect_exit_code(0).expect_stdout(util.zeroth_line_starts_with(u"usage:"))

    def test_unsupported_command(self):
        cmd = [self.script, "hehe"]
        msg = u"chunkify.py: error: argument command: invalid choice:"
        util.run_cmd(self, cmd).expect_exit_code(2).expect_stderr(util.nth_line_starts_with(msg, 1))

    @parameterized.expand([
        [[], (182, 500, 4), -2.8844583, 14.225174, -0.254353493452, "0"],
        [["--normalisation", "per-read"], (182, 500, 4), -2.8844583, 14.225174, -0.254353493452, "1"],
        [["--normalisation", "per-chunk"], (182, 500, 4), -4.1303601265, 12.2556829453, -0.249717712402, "2"],
    ])
    def test_chunkify_with_identity_with_normalisation(self, options, chunks_shape, min_value, max_value, median_value,
                                                       subdir):
        test_work_dir = self.work_dir(os.path.join("test_chunkify_with_identity_with_normalisation", subdir))

        strand_input_list = os.path.join(self.data_dir, "identity", "na12878_train.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "identity", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        output_file_name = os.path.join(test_work_dir, "output.hdf5")
        open(output_file_name, "w").close()

        cmd = [self.script, "identity", "--chunk_len", "500", "--kmer_len", "5",
               "--section", "template", "--input_strand_list", strand_input_list,
               reads_dir, output_file_name] + options

        util.run_cmd(self, cmd).expect_exit_code(1)

        util.run_cmd(self, cmd + ['--overwrite']).expect_exit_code(0)

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
        ['remap', 'Segment_Linear', [],
            (33, 500, 4), -2.7013657093, 12.7536773682, -0.238673046231, "0"],
        ['remap', 'Segment_Linear', ["--normalisation", "per-read"],
            (33, 500, 4), -2.7013657093, 12.7536773682, -0.238673046231, "1"],
        ['remap', 'Segment_Linear', ["--normalisation", "per-chunk"],
            (33, 500, 4), -2.88131427765, 11.0136013031, -0.238405257463, "2"],
        ['remap3', 'Segmentation', ["--normalisation", "per-chunk"],
            (17, 500, 4), -3.07923054695, 11.3292417526, -0.212918192148, "3"],
    ])
    def test_chunkify_with_remap_with_normalisation(self, subdir, segmentation, options, chunks_shape, min_value,
                                                    max_value, median_value, test_id):
        test_work_dir = self.work_dir(os.path.join("test_chunkify_with_remap_with_normalisation", test_id))

        strand_input_list = os.path.join(self.data_dir, subdir, "output_strand_list.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, subdir, "reads")
        self.assertTrue(os.path.exists(reads_dir))

        model_file = os.path.join(self.data_dir, subdir, "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        reference_file = os.path.join(self.data_dir, subdir, "reference.fa")
        self.assertTrue(os.path.exists(reference_file))

        output_strand_list = os.path.join(test_work_dir, "output_strand_list.txt")
        open(output_strand_list, 'w').close()

        output_file_name = os.path.join(test_work_dir, "output.hdf5")
        open(output_file_name, 'w').close()

        cmd = [self.script, "remap", "--segmentation", segmentation, "--trim", "200", "200",
               "--chunk_len", "500", "--kmer_len", "5", "--section", "template",
               "--output_strand_list",
               output_strand_list, reads_dir, output_file_name, model_file, reference_file] + options

        util.run_cmd(self, cmd).expect_exit_code(1)

        os.remove(output_file_name)

        util.run_cmd(self, cmd).expect_exit_code(2)

        util.run_cmd(self, cmd + ['--overwrite']).expect_exit_code(0)

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
        os.remove(output_strand_list)

    @parameterized.expand([
        [[], (57, 2000, 1), -2.82059764862, 3.37245368958, 0.0, "0"],
        [["--normalisation", "per-read"], (57, 2000, 1), -2.82059764862, 3.37245368958, 0.0, "0"],
        [["--normalisation", "per-chunk"], (57, 2000, 1), -12.6804265976, 22.79778862, 0.0, "2"]
    ])
    def test_chunkify_with_raw_remap_with_normalisation(self, options, chunks_shape, min_value, max_value, median_value,
                                                        subdir):
        test_work_dir = self.work_dir(os.path.join("test_chunkify_with_raw_remap_with_normalisation", subdir))

        reads_dir = os.path.join(self.data_dir, "raw_remap", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        model_file = os.path.join(self.data_dir, "raw_remap", "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        reference_file = os.path.join(self.data_dir, "raw_remap", "reference.fa")
        self.assertTrue(os.path.exists(reference_file))

        output_strand_list = os.path.join(test_work_dir, "output_strand_list.txt")
        open(output_strand_list, 'w').close()

        output_file_name = os.path.join(test_work_dir, "output.hdf5")
        open(output_file_name, 'w').close()

        cmd = [self.script, "raw_remap", "--overwrite", "--downsample", "5",
               reads_dir, output_file_name, model_file, reference_file,
               '--output_strand_list', output_strand_list] + options

        util.run_cmd(self, cmd).expect_exit_code(0)

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
        os.remove(output_strand_list)

    def test_chunkify_with_remap_no_results_due_to_missing_reference(self):
        test_work_dir = self.work_dir("test_chunkify_with_remap_no_results_due_to_missing_reference")

        strand_input_list = os.path.join(self.data_dir, "remap", "output_strand_list.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "identity", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        model_file = os.path.join(self.data_dir, "remap", "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        reference_file = os.path.join(self.data_dir, "remap", "reference.fa")
        self.assertTrue(os.path.exists(reference_file))

        output_strand_list = os.path.join(test_work_dir, "output_strand_list.txt")
        if os.path.exists(output_strand_list):
            os.remove(output_strand_list)

        output_file_name = os.path.join(test_work_dir, "output.hdf5")
        if os.path.exists(output_file_name):
            os.remove(output_file_name)

        cmd = [self.script, "remap", "--segmentation", "Segment_Linear", "--trim", "200", "200",
               "--chunk_len", "500", "--kmer_len", "5", "--section", "template",
               "--input_strand_list", strand_input_list, "--output_strand_list", output_strand_list,
               reads_dir, output_file_name, model_file, reference_file]

        util.run_cmd(self, cmd).expect_exit_code(1).expect_stderr(
            util.last_line_starts_with(u"no chunks were produced"))

        self.assertTrue(not os.path.exists(output_file_name))
        self.assertTrue(not os.path.exists(output_strand_list))

    @parameterized.expand([
        [495, 540, 25, 20, 0, "0"],
        [496, 540, 25, 20, 1, "1"],
        [495, 541, 25, 20, 1, "2"],
        [495, 540, 26, 20, 1, "3"],
        [495, 540, 25, 21, 1, "4"],
    ])
    def test_chunkify_with_remap_no_results_due_to_length(self, chunk_len, min_length, trim_left, trim_right,
                                                          exit_code, subdir):
        test_work_dir = self.work_dir(os.path.join("test_chunkify_with_remap_no_results_due_to_length", subdir))

        strand_input_list = os.path.join(self.data_dir, "remap2", "strand_input_list.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "remap2", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        model_file = os.path.join(self.data_dir, "remap2", "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        reference_file = os.path.join(self.data_dir, "remap2", "reference.fa")
        self.assertTrue(os.path.exists(reference_file))

        output_strand_list = os.path.join(test_work_dir, "output_strand_list.txt")
        if os.path.exists(output_strand_list):
            os.remove(output_strand_list)

        output_file_name = os.path.join(test_work_dir, "output.hdf5")
        if os.path.exists(output_file_name):
            os.remove(output_file_name)

        cmd = [self.script, "remap", "--segmentation", "Segment_Linear",
               "--trim", str(trim_left), str(trim_right), "--chunk_len", str(chunk_len),
               "--kmer_len", "5", "--section", "template", "--input_strand_list", strand_input_list,
               "--output_strand_list", output_strand_list, "--min_length", str(min_length),
               reads_dir, output_file_name, model_file, reference_file]

        expectation = util.run_cmd(self, cmd).expect_exit_code(exit_code)

        if exit_code != 0:
            expectation.expect_stderr(util.last_line_starts_with(u"no chunks were produced"))

            self.assertTrue(not os.path.exists(output_file_name))
            self.assertTrue(not os.path.exists(output_strand_list))

    @parameterized.expand([
        [300, 360, 40, 20, 0, "0"],
        [301, 360, 40, 20, 1, "1"],
        [300, 361, 40, 20, 1, "2"],
        [300, 360, 41, 20, 1, "3"],
        [300, 360, 40, 21, 1, "4"],
    ])
    def test_chunkify_with_identity_no_results_due_to_length(self, chunk_len, min_length, trim_left, trim_right,
                                                             exit_code, subdir):
        test_work_dir = self.work_dir(os.path.join("test_chunkify_with_identity_no_results_due_to_length", subdir))

        strand_input_list = os.path.join(self.data_dir, "remap2", "strand_input_list.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "remap2", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        output_file_name = os.path.join(test_work_dir, "output.hdf5")
        if os.path.exists(output_file_name):
            os.remove(output_file_name)

        cmd = [self.script, "identity", "--trim", str(trim_left), str(trim_right), "--chunk_len", str(chunk_len),
               "--kmer_len", "5", "--section", "template", "--input_strand_list", strand_input_list,
               "--min_length", str(min_length), reads_dir, output_file_name]

        expectation = util.run_cmd(self, cmd).expect_exit_code(exit_code)

        if exit_code != 0:
            expectation.expect_stderr(util.last_line_starts_with(u"no chunks were produced"))

            self.assertTrue(not os.path.exists(output_file_name))
