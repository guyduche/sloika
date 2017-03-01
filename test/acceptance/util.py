from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import os
import itertools

from subprocess import Popen, PIPE


class Result(object):

    def __init__(self, test_case, cmd, cwd, return_code, stdout, stderr):
        self.test_case = test_case
        self.cmd = cmd
        self.cwd = cwd
        self._return_code = return_code
        self._stdout = stdout.strip('\n').split('\n')
        self._stderr = stderr.strip('\n').split('\n')

    def __repr__(self):
        L = ['\n\tCommand: {}'.format(' '.join(self.cmd))]
        if self.cwd:
            L.append('\n\tCwd: {}'.format(self.cwd))

        if self._return_code:
            L.append('\tCommand exit code: %s' % self._return_code)

        if self._stdout:
            L.append('\n\tCommand output:')
            for line in self._stdout:
                L.append("\t\t{}".format(line))

        if self._stderr:
            L.append('\n\tCommand error output:')
            for line in self._stderr:
                L.append("\t\t{}".format(line))

        return '\n'.join(L)

    def return_code(self, expected_return_code):
        msg = "expected return code %s but got %s in: %s" % (expected_return_code, self._return_code, self)
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
    env_with_theano_flags = os.environ.copy()
    base_compiledir = os.path.join(test_case.work_dir, '.theano')
    env_with_theano_flags["THEANO_FLAGS"] = "base_compiledir={},".format(
        base_compiledir) + os.environ["THEANO_FLAGS_FOR_ACCTEST"]

    proc = Popen(cmd, env=env_with_theano_flags, stdout=PIPE, stderr=PIPE, cwd=cwd)
    stdout, stderr = proc.communicate(None)

    return_code = proc.returncode
    stdout = stdout.decode('UTF-8')
    stderr = stderr.decode('UTF-8')

    return Result(test_case, cmd, cwd, return_code, stdout, stderr)


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def maybe_create_dir(directory_name):
    '''
    Create a directory if it does not exist already.
    In Python 2.7 OSError is thrown if directory does not exist or permissions are insufficient.
    In Python 3 more specific exceptions are thrown.
    '''

    try:
        os.makedirs(directory_name)
    except OSError:
        if os.path.exists(directory_name) and os.path.isdir(directory_name):
            pass
        else:
            raise


def nth_line_starts_with(prefix, n):
    def f(L):
        try:
            return L[n].startswith(prefix)
        except IndexError:
            return False
    return f


def zeroth_line_starts_with(prefix):
    return nth_line_starts_with(prefix, 0)


def last_line_starts_with(prefix):
    return nth_line_starts_with(prefix, -1)

if __name__ == '__main__':
    assert not zeroth_line_starts_with('a')([])
    assert zeroth_line_starts_with('a')(['a'])
    assert zeroth_line_starts_with('a')(['a', 'a'])
    assert zeroth_line_starts_with('a')(['a', 'b'])
    assert not zeroth_line_starts_with('a')(['b', 'a'])

    assert not last_line_starts_with('a')([])
    assert last_line_starts_with('a')(['a'])
    assert last_line_starts_with('a')(['a', 'a'])
    assert not last_line_starts_with('a')(['a', 'b'])
    assert last_line_starts_with('a')(['b', 'a'])
