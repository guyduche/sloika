from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

from nose_parameterized import parameterized
import os
import shutil
import unittest

from util import run_cmd, last_line_starts_with, maybe_create_dir, zeroth_line_starts_with


class AcceptanceTest(unittest.TestCase):

    maxDiff = None

    @classmethod
    def setUpClass(self):
        self.test_directory = os.path.splitext(__file__)[0]
        self.test_name = os.path.basename(self.test_directory)
        self.script = os.path.join(os.environ["BIN_DIR"], "basecall_network")

        self.work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], self.test_name)
        maybe_create_dir(self.work_dir)

        self.data_dir = os.path.join(os.environ["DATA_DIR"], self.test_name)

    def test_usage(self):
        cmd = [self.script]
        run_cmd(self, cmd).return_code(2).stderr(zeroth_line_starts_with(u"usage"))

    def test_raw_iteration_failure_on_files_with_no_raw_data(self):
        model_file = os.path.join(self.data_dir, "raw_model_1pt2_cpu.pkl")
        self.assertTrue(os.path.exists(model_file))

        reads_dir = os.path.join(self.data_dir, "no_raw", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        cmd = [self.script, "raw", model_file, reads_dir]
        run_cmd(self, cmd).return_code(0).stderr(last_line_starts_with(u"Called 0 bases"))

    @parameterized.expand([
        [[]],
        [['--open_pore_fraction', '0']],
    ])
    def test_basecall_network_raw(self, options):
        model_file = os.path.join(self.data_dir, "raw_model_1pt2_cpu.pkl")
        self.assertTrue(os.path.exists(model_file))

        reads_dir = os.path.join(self.data_dir, "raw", "dataset1", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        expected_output_file = os.path.join(self.data_dir, "raw", "dataset1", "output.txt")
        self.assertTrue(os.path.exists(expected_output_file))
        expected_output = open(expected_output_file, 'r').read().splitlines()

        cmd = [self.script, "raw", model_file, reads_dir] + options
        run_cmd(self, cmd).return_code(0).stdoutEquals(expected_output)

    @parameterized.expand([
        [[]],
        [['--trim', '50', '1']],
    ])
    def test_basecall_network_events(self, options):
        model_file = os.path.join(self.data_dir, "events_model_cpu.pkl")
        self.assertTrue(os.path.exists(model_file))

        reads_dir = os.path.join(self.data_dir, "events", "dataset1", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        expected_output_file = os.path.join(self.data_dir, "events", "dataset1", "output.txt")
        self.assertTrue(os.path.exists(expected_output_file))
        expected_output = open(expected_output_file, 'r').read().splitlines()

        cmd = [self.script, "events", "--segmentation", "Segment_Linear", model_file, reads_dir] + options
        run_cmd(self, cmd).return_code(0).stdoutEquals(expected_output)
