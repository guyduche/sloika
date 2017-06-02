import h5py
from nose_parameterized import parameterized
import os
import shutil
import sys
import unittest

import util


class AcceptanceTest(unittest.TestCase):

    known_commands = ["events", "raw"]

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
        msg = "train_network.py: error: the following arguments are required: command"
        util.run_cmd(self, cmd).expect_exit_code(2).expect_stderr(util.last_line_starts_with(msg))

    @parameterized.expand(known_commands)
    def test_commands_usage(self, command_name):
        cmd = [self.script, command_name]
        msg = "train_network.py {}: error: the following arguments are required: model, output, input".format(
                command_name)
        util.run_cmd(self, cmd).expect_exit_code(2).expect_stderr(util.last_line_starts_with(msg))

    @parameterized.expand([
        ["0"],
    ])
    def test_baseline_lstm_training(self, subdir):
        test_work_dir = self.work_dir(os.path.join("test_baseline_lstm_training", subdir))

        strand_input_list = os.path.join(self.data_dir, "events", "na12878_train.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "events", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        hdf5_file = os.path.join(test_work_dir, "output.hdf5")

        prepare_cmd = [self.chunkify_script, "identity", "--chunk_len", "500", "--kmer_len", "5",
                       "--section", "template", "--input_strand_list", strand_input_list,
                       reads_dir, hdf5_file, "--overwrite"]

        util.run_cmd(self, prepare_cmd).expect_exit_code(0)

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

        train_cmd = [self.script, "events", "--batch_size", "100", "--niteration", "1", "--save_every", "1",
                     "--lrdecay", "5000", "--bad", model, output_directory, hdf5_file]

        util.run_cmd(self, train_cmd).expect_exit_code(0)

        self.assertTrue(os.path.exists(output_directory))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model_checkpoint_00000.pkl")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model_checkpoint_00001.pkl")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model_final.pkl")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model.log")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model.py")))

    @parameterized.expand([
        ["small_ch3000_lt0.825_simple2.hdf5", "0"],
        ["small_ch3000_lt0.582_simple5.hdf5", "1"],
    ])
    def test_baseline_raw_gru_training(self, hdf5_file_name, subdir):
        test_work_dir = self.work_dir(os.path.join("test_baseline_raw_gru_training", subdir))

        model = os.path.join(self.models_dir, "baseline_raw_gru.py")
        self.assertTrue(os.path.exists(model))

        output_directory = os.path.join(test_work_dir, "training_output")
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)

        hdf5_file = os.path.join(self.data_dir, "raw", hdf5_file_name)
        self.assertTrue(os.path.exists(hdf5_file))

        train_cmd = [self.script, "raw", "--batch_size", "50", "--niteration", "1", "--save_every", "1",
                     "--lrdecay", "1000", "--winlen", "11",
                     "--chunk_len_range", "0.1", "0.1", model, output_directory, hdf5_file]

        util.run_cmd(self, train_cmd).expect_exit_code(0)

        self.assertTrue(os.path.exists(output_directory))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model_checkpoint_00000.pkl")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model_checkpoint_00001.pkl")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model_final.pkl")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model.log")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model.py")))
