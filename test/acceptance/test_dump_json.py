from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

from nose_parameterized import parameterized
import os
import shutil
import sys
import tempfile
import unittest

from util import run_cmd, maybe_create_dir, zeroth_line_starts_with, last_line_starts_with


class AcceptanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_directory = os.path.splitext(__file__)[0]
        self.test_name = os.path.basename(self.test_directory)
        self.script = os.path.join(os.environ["BIN_DIR"], "dump_json.py")

        self.work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], self.test_name)
        maybe_create_dir(self.work_dir)

        self.data_dir = os.path.join(os.environ["DATA_DIR"], self.test_name)

    def test_usage(self):
        cmd = [self.script]
        run_cmd(self, cmd).return_code(2).stderr(zeroth_line_starts_with(u"usage"))

    @parameterized.expand([
        [[], "model_py{}.json"],
        [["--params"], "model_py{}.json"],
        [["--no-params"], "model_without_params_py{}.json"]
    ])
    def test_dump_to_stdout(self, options, reference_dump_file_name_template):
        model_file = os.path.join(self.data_dir, "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        reference_dump_path = os.path.join(
            self.data_dir, reference_dump_file_name_template.format(sys.version_info.major))
        self.assertTrue(os.path.exists(reference_dump_path))

        reference_dump = open(reference_dump_path, 'r').read().splitlines()

        cmd = [self.script, model_file] + options
        run_cmd(self, cmd).return_code(0).stdoutEquals(reference_dump)

    @parameterized.expand([
        [[], "model_py{}.json"],
        [["--params"], "model_py{}.json"],
        [["--no-params"], "model_without_params_py{}.json"]
    ])
    def test_dump_to_a_file(self, options, reference_dump_file_name_template):
        model_file = os.path.join(self.data_dir, "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        reference_dump_path = os.path.join(
            self.data_dir, reference_dump_file_name_template.format(sys.version_info.major))
        self.assertTrue(os.path.exists(reference_dump_path))

        reference_dump = open(reference_dump_path, 'r').read().splitlines()

        with tempfile.NamedTemporaryFile(dir=self.work_dir, suffix=".json", delete=False) as fh:
            out_file = fh.name

        cmd = [self.script, model_file, "--out_file", out_file] + options
        error_message = "RuntimeError: File/path for 'out_file' exists, {}".format(out_file)
        run_cmd(self, cmd).return_code(1).stderr(last_line_starts_with(error_message))

        os.remove(out_file)

        info_message = "Writing to file:  {}".format(out_file)
        run_cmd(self, cmd).return_code(0).stdout(lambda o: o == [info_message])

        self.assertTrue(os.path.exists(out_file))
        dump = open(out_file, 'r').read().splitlines()

        self.assertEqual(dump, reference_dump)
