#!/usr/bin/env python
import argparse
from itertools import islice
import multiprocessing
import os
import sqlite3
import subprocess
import time
from untangled.cmdargs import FileExists, Maybe, NonNegative, Positive

_PENDING = 0
_RUNNING = 1
_SUCCESS = 2
_FAILURE = 3
_SUSPEND = 4

sloika_gitdir = "/home/ubuntu/git/sloika"

parser = argparse.ArgumentParser(
    description = 'server for model training',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--limit', metavar='jobs', default=None, type=Maybe(Positive(int)),
    help='Maximum number of jobs')
parser.add_argument('--sleep', metavar='seconds', default=30, type=NonNegative(int),
    help='Time between polling database')
parser.add_argument('database', metavar='file.db', action=FileExists,
    help='')


def get_git_commit(gitdir):
    return subprocess.check_output(
              'cd {} && git log --pretty=format:"%H" -1'.format(gitdir), shell=True
          ).rstrip()


def create_jobs(dbname, sleep=30, limit=None):
    njobs = 0
    with sqlite3.connect(dbname) as conn:
        conn.row_factory = sqlite3.Row
        while limit is None or njobs < limit:
            c = conn.cursor()
            c.execute('select * from runs where status = ? order by priority limit 1', (_PENDING,))
            res = c.fetchone()
            if res is not None:
                runid = res["runid"]
                c.execute('update runs set status = ? where runid = ?', (_RUNNING, runid))
                conn.commit()
                njobs += 1
                yield dict(res)
            else:
                #  Yield dummy jobs so queue of finished jobs can empty
                yield None
                time.sleep(sleep)


def imap_unordered(pool, f, iterable):
    nproc = pool._processes

    #  Fill job array with at most nproc processes
    jobs = [pool.apply_async(f, (args,)) for args in islice(iterable, nproc)]

    #  Repeatedly scan through job array looking for new jobs
    while any(jobs):
        for i in xrange(len(jobs)):
            if jobs[i] is not None and jobs[i].ready():
                res = jobs[i].get()
                try:
                    next_job = iterable.next()
                    jobs[i] = pool.apply_async(f, (next_job,))
                except StopIteration:
                    jobs[i] = None

                yield res


def run_job(args):
    if args is None:
        return None

    # Theano flags
    pid = int(multiprocessing.current_process().name.split('-')[-1])
    gpu = (pid - 1) // 2
    env = os.environ.copy()
    env['THEANO_FLAGS'] = 'floatX=float32,warn_float64=warn,optimizer=fast_run,nvcc.fastmath=True,device=gpu{},scan.allow_gc=False,lib.cnmem=0.3'.format(gpu)

    # arglist for training
    arglist = [os.path.join(sloika_gitdir,"bin/train_network.py"),
               "--window", "3",
               "--bad",
               args["model"],
               args["output_directory"],
               args["training_data"]
               ]
    if args.transducer > 0:
        arglist.append("--transducer")
    else:
        arglist.append("--no-transducer")

    proc = subprocess.Popen(arglist, env=env)
    proc.wait()
    returncode = proc.returncode

    if returncode == 0 and args["validation_data"] is not None:
        # arglist for validation
        final_model = os.path.join(args["output_directory"], "model_final.pkl")
        arglist = [os.path.join(sloika_gitdir,"bin/validate_network.py"),
                   "--bad",
                   final_model,
                   args["validation_data"]
                   ]
        if args.transducer > 0:
            arglist.append("--transducer")
        else:
            arglist.append("--no-transducer")

        with open(os.path.join(args["output_directory"], "model_final.validate"), "w") as fh:
            proc = subprocess.Popen(arglist, env=env, stdout=fh)
            proc.wait()
        returncode2 = proc.returncode

    return args["runid"], returncode


if __name__ == '__main__':
    args = parser.parse_args()

    jobs = create_jobs(args.database, sleep=args.sleep, limit=args.limit)
    pool = multiprocessing.Pool(8)
    for res in imap_unordered(pool, run_job, jobs):
        if res is None:
            continue
        runid, returncode = res
        status = _SUCCESS if returncode == 0 else _FAILURE
        commit = get_git_commit(sloika_gitdir)
        with sqlite3.connect(args.database) as conn:
            c = conn.cursor()
            c.execute('update runs set status = ?, sloika_commit = ? where runid = ?', (status, commit, runid))
    pool.close()
    pool.join()
