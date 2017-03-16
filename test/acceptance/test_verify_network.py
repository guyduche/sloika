from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import glob
import os
import numpy as np
import shutil
import unittest

from nose_parameterized import parameterized

from util import run_cmd, maybe_create_dir, zeroth_line_starts_with


class AcceptanceTest(unittest.TestCase):
    model_files = [[x] for x in glob.glob(os.path.join(os.environ["ROOT_DIR"], "models/*.py"))]
    events_model_files = filter(lambda x: "raw" not in x[0], model_files)
    raw_model_files = filter(lambda x: "raw" in x[0], model_files)

    @classmethod
    def setUpClass(self):
        self.test_directory = os.path.splitext(__file__)[0]
        self.test_name = os.path.basename(self.test_directory)
        self.script = os.path.join(os.environ["BIN_DIR"], "verify_network.py")

        self.work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], self.test_name)
        maybe_create_dir(self.work_dir)

    def test_usage(self):
        cmd = [self.script]
        run_cmd(self, cmd).return_code(2).stderr(zeroth_line_starts_with(u"usage"))

    def test_number_of_models(self):
        '''
        Check we've found at least one model
        '''
        self.assertTrue(len(self.model_files) > 0)

    @parameterized.expand(events_model_files)
    def test_sequence_events(self, model_file):
        cmd = [self.script, "--kmer", "5", model_file]
        run_cmd(self, cmd).return_code(0)

    @parameterized.expand(raw_model_files)
    def test_sequence_raw(self, model_file):
        stride = str(np.random.randint(1, 10))
        cmd = [self.script, "--kmer", "5", "--stride", stride, model_file]
        run_cmd(self, cmd).return_code(0)
