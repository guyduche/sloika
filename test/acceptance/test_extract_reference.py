from nose_parameterized import parameterized
import os
import unittest

import util


def ordered(L):
    assert len(L) % 2 == 0, "Number of lines in fasta file must be even"
    pairs = [L[i:i + 2] for i in range(0, len(L), 2)]
    return list(sorted(pairs))


class AcceptanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        testset_directory = os.path.splitext(__file__)[0]
        testset_name = os.path.basename(testset_directory)

        self.data_dir = os.path.join(os.environ["DATA_DIR"], testset_name)

        self.testset_work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], testset_name)

        self.script = os.path.join(os.environ["BIN_DIR"], "extract_reference.py")

    def work_dir(self, test_name):
        directory = os.path.join(self.testset_work_dir, test_name)
        util.maybe_create_dir(directory)
        return directory

    def test_commands(self):
        cmd = [self.script, "--help"]
        util.run_cmd(self, cmd).expect_exit_code(0).expect_stdout(util.zeroth_line_starts_with(u"usage:"))

    @parameterized.expand([
        [[], 'output.fa', None, '0'],
        [['--section', 'template'], 'output.fa', None, '1'],
        [[], 'output4.fa', 'strand_list.txt', '3'],
    ])
    def test_functionality(self, options, reference_fasta, input_strand, subdir):
        test_work_dir = self.work_dir(os.path.join("test_functionality", subdir))

        if input_strand is not None:
            input_strand_list = os.path.join(self.data_dir, input_strand)
            self.assertTrue(os.path.exists(input_strand_list))
            options = options + ['--input_strand_list', input_strand_list]

        reads_dir = os.path.join(self.data_dir, "reads")
        self.assertTrue(os.path.exists(reads_dir))

        reference_output_file_name = os.path.join(self.data_dir, reference_fasta)
        self.assertTrue(os.path.exists(reference_output_file_name))

        output_file_name = os.path.join(test_work_dir, "output.fa")
        open(output_file_name, "w").close()

        cmd = [self.script, reads_dir, output_file_name] + options

        util.run_cmd(self, cmd).expect_exit_code(1)

        util.run_cmd(self, cmd + ['--overwrite']).expect_exit_code(0)

        self.assertTrue(os.path.exists(output_file_name))

        output = ordered(open(output_file_name, 'r').readlines())
        reference_output = ordered(open(reference_output_file_name, 'r').readlines())

        self.assertEqual(output, reference_output)

    @parameterized.expand([
        [[], 1, '0'],
        [[], 3, '1'],
    ])
    def test_functionality_with_limit(self, options, limit, subdir):
        test_work_dir = self.work_dir(os.path.join("test_functionality_with_limit", subdir))

        reads_dir = os.path.join(self.data_dir, "reads")
        self.assertTrue(os.path.exists(reads_dir))

        output_file_name = os.path.join(test_work_dir, "output.fa")

        cmd = [self.script, reads_dir, output_file_name, "--limit", str(limit), "--overwrite"] + options

        util.run_cmd(self, cmd).expect_exit_code(0)

        self.assertTrue(os.path.exists(output_file_name))

        output = ordered(open(output_file_name, 'r').readlines())

        self.assertEqual(len(output), limit)

    @parameterized.expand([
        [[], '0'],
    ])
    def test_can_cope_with_files_with_no_reference(self, options, subdir):
        test_work_dir = self.work_dir(os.path.join("test_can_cope_with_files_with_no_reference", subdir))

        reads_dir = os.path.join(self.data_dir, "reads_without_ref")
        self.assertTrue(os.path.exists(reads_dir))

        output_file_name = os.path.join(test_work_dir, "output.fa")

        cmd = [self.script, reads_dir, output_file_name, "--overwrite"] + options

        util.run_cmd(self, cmd).expect_exit_code(0)

        self.assertTrue(os.path.exists(output_file_name))

        output = ordered(open(output_file_name, 'r').readlines())

        self.assertEqual(len(output), 0)
