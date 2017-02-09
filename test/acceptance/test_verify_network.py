import unittest
import os
import shutil
import glob

from nose_parameterized import parameterized

from utils import run_cmd, maybe_create_dir, drop_info

class AcceptanceTest(unittest.TestCase):
    model_files = map(lambda x: [x], glob.glob(os.path.join(os.environ["ROOT_DIR"], "models/*.py")))

    @classmethod
    def setUpClass(self):
        self.test_directory = os.path.splitext(__file__)[0]
        self.test_name = os.path.basename(self.test_directory)
        self.script = os.path.join(os.environ["BIN_DIR"], "verify_network.py")

        self.work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], self.test_name)
        maybe_create_dir(self.work_dir)

    def test_usage(self):
        cmd = [self.script]
        run_cmd(self, cmd).return_code(2).stderr(lambda o: drop_info(o)[0].startswith(u"usage"))

    def test_number_of_models(self):
        '''
        Check we've found at least one model
        '''
        self.assertTrue(len(self.model_files) > 0)

    @parameterized.expand(model_files)
    def test_sequence(self, model_file):
        cmd = [self.script, "--kmer", "5", model_file]
        run_cmd(self, cmd).return_code(0)
