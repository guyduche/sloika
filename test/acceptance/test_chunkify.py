import unittest
import os
import shutil
import h5py
import shutil
import difflib

import numpy as np

from distutils import dir_util
from nose_parameterized import parameterized


from utils import run_cmd, is_close, maybe_create_dir, drop_info


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

    def test_commands(self):
        cmd = [self.script]
        run_cmd(self, cmd).return_code(0).stdout(lambda o: drop_info(o).startswith(u"Available commands:"))

    @parameterized.expand(known_commands)
    def test_usage(self, command_name):
        cmd = [self.script, command_name]
        run_cmd(self, cmd).return_code(2).stderr(lambda o: drop_info(o).startswith(u"usage:"))

    def test_unsupported_command(self):
        cmd = [self.script, "hehe"]
        run_cmd(self, cmd).return_code(1).stdout(lambda o: drop_info(o).startswith(u"Unsupported command 'hehe'"))

    def test_identity(self):
        strand_input_list = os.path.join(self.data_dir, "identity", "na12878_train.txt")
        self.assertTrue(os.path.exists(strand_input_list))

        reads_dir = os.path.join(self.data_dir, "identity", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        output_file = os.path.join(self.work_dir, "dataset_train.hdf5")
        if os.path.exists(output_file):
            os.remove(output_file)

        cmd = [self.script, "identity", "--use_scaled", "--chunk", "500", "--kmer", "5",
               "--section", "template", "--strand_list", strand_input_list,
               reads_dir, output_file]
        run_cmd(self, cmd).return_code(0)

        with h5py.File(output_file, 'r') as fh:
            top_level_items = []
            for item in fh:
                top_level_items.append(item)
            top_level_items.sort()
            self.assertEqual(top_level_items, [u'bad', u'chunks', u'labels', u'weights'])

            self.assertEqual(fh['chunks'].shape, (182, 500, 4))
            chunks = fh['chunks'][:]
            self.assertTrue(is_close(chunks.min(), -2.8844583, 1e-5))
            self.assertTrue(is_close(chunks.max(), 14.225174, 1e-5))

    def test_remap(self):
        strand_output_list = os.path.join(self.work_dir, "strand_output_list.txt")
        if os.path.exists(strand_output_list):
            os.remove(strand_output_list)

        model_file = os.path.join(self.data_dir, "remap", "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        reference_file = os.path.join(self.data_dir, "remap", "reference.fa")
        self.assertTrue(os.path.exists(reference_file))

        reads_dir = os.path.join(self.data_dir, "remap", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        temporary_reads_dir = os.path.join(self.work_dir, "reads")
        if os.path.exists(temporary_reads_dir):
            shutil.rmtree(temporary_reads_dir)
        dir_util.copy_tree(reads_dir, temporary_reads_dir)

        reference_strand_output_list = os.path.join(self.data_dir, "remap", "strand_output_list.txt")
        self.assertTrue(os.path.exists(reference_strand_output_list))

        cmd = [self.script, "remap", "--strand_output_list", strand_output_list, model_file, reference_file, temporary_reads_dir]

        run_cmd(self, cmd).return_code(0)

        self.assertTrue(os.path.exists(strand_output_list))
        strand_output_list_contents = open(strand_output_list, "r").readlines()
        reference_strand_output_list_contents = open(reference_strand_output_list, "r").readlines()
        diff = list(difflib.context_diff( reference_strand_output_list_contents, strand_output_list_contents, "generated", "reference"))
        if len(diff) != 0:
            print(''.join(diff))
            self.assertTrue(reference_strand_output_list_contents == strand_output_list_contents)
