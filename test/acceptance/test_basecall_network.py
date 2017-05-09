from nose_parameterized import parameterized
import os
import unittest

import util


class AcceptanceTest(unittest.TestCase):

    maxDiff = None

    @classmethod
    def setUpClass(self):
        test_directory = os.path.splitext(__file__)[0]
        testset_name = os.path.basename(test_directory)

        self.script = os.path.join(os.environ["BIN_DIR"], "basecall_network")

        self.testset_work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], testset_name)

        self.data_dir = os.path.join(os.environ["DATA_DIR"], testset_name)

    def test_usage(self):
        cmd = [self.script]
        util.run_cmd(self, cmd).expect_exit_code(2).expect_stderr(util.zeroth_line_starts_with(u"usage"))

    def test_raw_iteration_failure_on_files_with_no_raw_data(self):
        model_file = os.path.join(self.data_dir, "raw_model_1pt2_cpu.pkl")
        self.assertTrue(os.path.exists(model_file))

        reads_dir = os.path.join(self.data_dir, "no_raw", "reads")
        self.assertTrue(os.path.exists(reads_dir))

        cmd = [self.script, "raw", model_file, reads_dir]
        util.run_cmd(self, cmd).expect_exit_code(0).expect_stderr(util.last_line_starts_with(u"Called 0 bases"))

    @parameterized.expand([
        [[]],
        [['--open_pore_fraction', '0']],
    ])
    def test_basecall_network_raw(self, options):
        model_file = os.path.join(self.data_dir, "raw_model_1pt2_cpu.pkl")
        self.assertTrue(os.path.exists(model_file))

        test_data_dir = os.path.join(self.data_dir, "raw", "dataset1")

        reads_dir = os.path.join(test_data_dir, "reads")
        self.assertTrue(os.path.exists(reads_dir))

        expected_output_file = os.path.join(test_data_dir, "output.txt")
        self.assertTrue(os.path.exists(expected_output_file))
        expected_output = open(expected_output_file, 'r').read().splitlines()

        cmd = [self.script, "raw", model_file, reads_dir] + options
        util.run_cmd(self, cmd).expect_exit_code(0).expect_stdout_equals(expected_output)

    @parameterized.expand([
        [[]],
        [['--trim', '50', '1']],
    ])
    def test_basecall_network_events(self, options):
        model_file = os.path.join(self.data_dir, "events_model_cpu.pkl")
        self.assertTrue(os.path.exists(model_file))

        test_data_dir = os.path.join(self.data_dir, "events", "dataset2")

        reads_dir = os.path.join(test_data_dir, "reads")
        self.assertTrue(os.path.exists(reads_dir))

        expected_output_file = os.path.join(test_data_dir, "output.txt")
        self.assertTrue(os.path.exists(expected_output_file))
        expected_output = open(expected_output_file, 'r').read().splitlines()

        cmd = [self.script, "events", "--segmentation", "Segment_Linear", model_file, reads_dir] + options
        util.run_cmd(self, cmd).expect_exit_code(0).expect_stdout_equals(expected_output)

    @parameterized.expand([
        [["--trans", "0.5", "0.4", "0.1", "--no-transducer"]],
    ])
    def test_basecall_network_events_with_non_default_trans(self, options):
        model_file = os.path.join(self.data_dir, "events_model_cpu.pkl")
        self.assertTrue(os.path.exists(model_file))

        test_data_dir = os.path.join(self.data_dir, "events", "dataset2")

        reads_dir = os.path.join(test_data_dir, "reads")
        self.assertTrue(os.path.exists(reads_dir))

        expected_output_file = os.path.join(test_data_dir, "output_no_transducer.txt")
        self.assertTrue(os.path.exists(expected_output_file))
        expected_output = open(expected_output_file, 'r').read().splitlines()

        cmd = [self.script, "events", "--segmentation", "Segment_Linear", model_file, reads_dir] + options
        util.run_cmd(self, cmd).expect_exit_code(0).expect_stdout_equals(expected_output)
