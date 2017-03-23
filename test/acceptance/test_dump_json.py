from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

from nose_parameterized import parameterized
import os
import sys
import unittest

import util


class AcceptanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        test_directory = os.path.splitext(__file__)[0]
        testset_name = os.path.basename(test_directory)

        self.testset_work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], testset_name)

        self.data_dir = os.path.join(os.environ["DATA_DIR"], testset_name)

        self.script = os.path.join(os.environ["BIN_DIR"], "dump_json.py")

    def work_dir(self, test_name):
        directory = os.path.join(self.testset_work_dir, test_name)
        util.maybe_create_dir(directory)
        return directory

    def test_usage(self):
        cmd = [self.script]
        util.run_cmd(self, cmd).return_code(2).stderr(util.zeroth_line_starts_with(u"usage"))

    @parameterized.expand([
        [[], "model_py{}.json"],
        [["--params"], "model_py{}.json"],
        [["--no-params"], "model_without_params_py{}.json"]
    ])
    def test_dump_to_stdout(self, options, reference_dump_file_name_template):
        model_file = os.path.join(self.data_dir, "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        majorMinor = "{}{}".format(sys.version_info.major, sys.version_info.minor)
        reference_dump_path = os.path.join(self.data_dir, reference_dump_file_name_template.format(majorMinor))
        self.assertTrue(os.path.exists(reference_dump_path))

        reference_dump = open(reference_dump_path, 'r').read().splitlines()

        cmd = [self.script, model_file] + options
        util.run_cmd(self, cmd).return_code(0).stdoutEquals(reference_dump)

    @parameterized.expand([
        [[], "model_py{}.json", "0"],
        [["--params"], "model_py{}.json", "1"],
        [["--no-params"], "model_without_params_py{}.json", "2"]
    ])
    def test_dump_to_a_file(self, options, reference_dump_file_name_template, subdir):
        test_work_dir = self.work_dir(os.path.join("test_dump_to_a_file", subdir))

        model_file = os.path.join(self.data_dir, "model.pkl")
        self.assertTrue(os.path.exists(model_file))

        majorMinor = "{}{}".format(sys.version_info.major, sys.version_info.minor)
        reference_dump_path = os.path.join(self.data_dir, reference_dump_file_name_template.format(majorMinor))
        self.assertTrue(os.path.exists(reference_dump_path))

        reference_dump = open(reference_dump_path, 'r').read().splitlines()

        output_file = os.path.join(test_work_dir, "output.json")

        cmd = [self.script, model_file, "--out_file", output_file] + options
        error_message = "RuntimeError: File/path for 'out_file' exists, {}".format(output_file)
        util.run_cmd(self, cmd).return_code(1).stderr(util.last_line_starts_with(error_message))

        os.remove(output_file)

        info_message = "Writing to file:  {}".format(output_file)
        util.run_cmd(self, cmd).return_code(0).stdout(lambda o: o == [info_message])

        self.assertTrue(os.path.exists(output_file))
        dump = open(output_file, 'r').read().splitlines()

        self.assertEqual(dump, reference_dump)
