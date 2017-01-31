import unittest
import os
import shutil
import h5py

import numpy as np

from utils import run_cmd, is_close, maybe_create_dir


class AcceptanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_directory = os.path.splitext(__file__)[0]
        self.test_name = os.path.basename(self.test_directory)
        self.script = os.path.join(os.environ["BIN_DIR"], "create_hdf5.py")

        self.work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], self.test_name)
	maybe_create_dir(self.work_dir)

        self.data_dir = os.path.join(os.environ["DATA_DIR"], self.test_name)

    def test_usage(self):
        cmd = [self.script]
        run_cmd(self, cmd).return_code(2).stderr(lambda o: o.startswith(u"usage"))

    def test_functionality(self):
        strand_list_file = os.path.join(self.data_dir, "na12878_train.txt")
        self.assertTrue(os.path.exists(strand_list_file))

        reads_dir = os.path.join(self.data_dir, "reads")
        self.assertTrue(os.path.exists(reads_dir))

        output_file = os.path.join(self.work_dir, "dataset_train.hdf5")
        if os.path.exists(output_file):
            os.remove(output_file)

        cmd = [self.script, "--use_scaled", "--chunk", "500", "--kmer", "5",
               "--section", "template", "--strand_list", strand_list_file,
               reads_dir, output_file]
        run_cmd(self, cmd).return_code(0)

        with h5py.File(output_file, 'r') as fh:
            top_level_items = []
            for item in fh:
                top_level_items.append(item)
            top_level_items.sort()
            self.assertEqual(top_level_items, [u'bad', u'centre', u'chunks', u'labels', u'rotation', u'weights'])

            self.assertEqual(fh['chunks'].shape, (182, 500, 4))
            chunks = fh['chunks'][:]
            self.assertTrue(is_close(chunks.min(), -2.8844583, 1e-5))
            self.assertTrue(is_close(chunks.max(), 14.225174, 1e-5))
