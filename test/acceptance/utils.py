from io import StringIO
from subprocess import Popen, PIPE


def run_cmd(cmd, cwd=None):
    print('\n\tCommand: {}'.format(' '.join(cmd)))
    if cwd:
        print('\n\tCwd: {}'.format(cwd))

    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=cwd)
    stdout, stderr = proc.communicate(StringIO())

    return_code = proc.returncode
    stdout = stdout.decode('UTF-8')
    stderr = stderr.decode('UTF-8')

    if stdout:
        print('\n\tCommand output:')
        for line in stdout.split('\n'):
            print("\t\t{}".format(line))

    if stderr:
        print('\n\tCommand error output:')
        for line in stderr.split('\n'):
            print("\t\t{}".format(line))

    if return_code:
        print('\tCommand exit code: %s' % return_code)

    return (return_code, stdout, stderr)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
