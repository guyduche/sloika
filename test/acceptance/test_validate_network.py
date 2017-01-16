import unittest
import os
import shutil

from utils import run_cmd


class AcceptanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_directory = os.path.splitext(__file__)[0]
        self.test_name = os.path.basename(self.test_directory)
        self.script = os.path.join( os.environ["BIN_DIR"], "verify_network.py" )

        self.work_dir = os.path.join(os.environ["ACCTEST_WORK_DIR"], self.test_name)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def test_usage(self):
        cmd = [self.script]
        return_code, stdout, stderr = run_cmd(cmd)
        self.assertEqual(return_code, 2)
        self.assertTrue(stderr.startswith(u"usage"))

