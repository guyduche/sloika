from io import StringIO
from subprocess import Popen, PIPE


class Result:

    def __init__(self, test_case, cmd, cwd, return_code, stdout, stderr):
        self.test_case = test_case
        self.cmd = cmd
        self.cwd = cwd
        self._return_code = return_code
        self._stdout = stdout
        self._stderr = stderr

    def __repr__(self):
        L = ['\n\tCommand: {}'.format(' '.join(self.cmd))]
        if self.cwd:
            L.append('\n\tCwd: {}'.format(self.cwd))

        if self._return_code:
            L.append('\tCommand exit code: %s' % self._return_code)

        if self._stdout:
            L.append('\n\tCommand output:')
            for line in self._stdout.split('\n'):
                L.append("\t\t{}".format(line))

        if self._stderr:
            L.append('\n\tCommand error output:')
            for line in self._stderr.split('\n'):
                L.append("\t\t{}".format(line))

        return '\n'.join(L)

    def return_code(self, expected_return_code):
        msg = "expected return code %s but got %s in: %s" % (self._return_code, expected_return_code, self)
        self.test_case.assertEqual(expected_return_code, self._return_code, msg)
        return self

    def stdout(self, f):
        msg = "expectation on stdout failed for: %s" % self
        self.test_case.assertTrue(f(self._stdout), msg)
        return self

    def stderr(self, f):
        msg = "expectation on stderr failed for: %s" % self
        self.test_case.assertTrue(f(self._stderr), msg)
        return self


def run_cmd(test_case, cmd, cwd=None):
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=cwd)
    stdout, stderr = proc.communicate(StringIO())

    return_code = proc.returncode
    stdout = stdout.decode('UTF-8')
    stderr = stderr.decode('UTF-8')

    return Result(test_case, cmd, cwd, return_code, stdout, stderr)


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
