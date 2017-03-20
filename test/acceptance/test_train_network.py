from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

import h5py
from nose_parameterized import parameterized
import os
import shutil
import unittest

import util


class AcceptanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        test_directory = os.path.splitext(__file__)[0]
        testset_name = os.path.basename(test_directory)

        self.testset_work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], testset_name)

        self.chunkify_script = os.path.join(os.environ["BIN_DIR"], "chunkify.py")

        self.script = os.path.join(os.environ["BIN_DIR"], "train_network.py")

        self.data_dir = os.path.join(os.environ["DATA_DIR"], testset_name)

        self.models_dir = os.path.join(os.environ["ROOT_DIR"], "models")

    def work_dir(self, test_name):
        directory = os.path.join(self.testset_work_dir, test_name)
        util.maybe_create_dir(directory)
        return directory

    def test_usage(self):
        cmd = [self.script]
        util.run_cmd(self, cmd).return_code(2).stderr(util.zeroth_line_starts_with(u"usage"))

    @parameterized.expand([
        [[]],
    ])
    def test_baseline_lstm_training(self, options):
        test_work_dir = self.work_dir("test_baseline_lstm_training")

        strand_input_list = os.path.join(self.data_dir, "events", "na12878_train.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "events", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        hdf5_file = util.create_file(test_work_dir, ".hdf5", False)

        prepare_cmd = [self.chunkify_script, "identity", "--chunk_len", "500", "--kmer_len", "5",
                       "--section", "template", "--input_strand_list", strand_input_list,
                       reads_dir, hdf5_file, "--overwrite"] + options

        util.run_cmd(self, prepare_cmd).return_code(0)

        with h5py.File(hdf5_file, 'r') as fh:
            top_level_items = []
            for item in fh:
                top_level_items.append(item)
            top_level_items.sort()
            self.assertEqual(top_level_items, [u'bad', u'chunks', u'labels', u'weights'])

        model = os.path.join(self.models_dir, "baseline_lstm.py")
        self.assertTrue(os.path.exists(model))

        output_directory = os.path.join(test_work_dir, "training_output")
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)

        train_cmd = [self.script, "--batch", "100", "--niteration", "1", "--save_every", "1", "--lrdecay", "5000",
                     "--bad", model, output_directory, hdf5_file]

        util.run_cmd(self, train_cmd).return_code(0)

        self.assertTrue(os.path.exists(output_directory))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model_checkpoint_00000.pkl")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model_checkpoint_00001.pkl")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model_final.pkl")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model.log")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model.py")))
