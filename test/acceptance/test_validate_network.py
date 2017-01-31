import unittest
import os
import shutil

from utils import run_cmd, maybe_create_dir


class AcceptanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_directory = os.path.splitext(__file__)[0]
        self.test_name = os.path.basename(self.test_directory)
        self.script = os.path.join(os.environ["BIN_DIR"], "verify_network.py")

        self.work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], self.test_name)
	maybe_create_dir(self.work_dir)

    def test_usage(self):
        cmd = [self.script]
        run_cmd(self, cmd).return_code(2).stderr(lambda o: o.startswith(u"usage"))
