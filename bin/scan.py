#!/usr/bin/env python
import argparse
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

parser = argparse.ArgumentParser(
    description = 'server for model training',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--limit', metavar='jobs', default=None, type=Maybe(Positive(int)),
    help='Maximum number of jobs')
parser.add_argument('--sleep', metavar='seconds', default=5, type=NonNegative(int),
    help='Time between polling database')
parser.add_argument('database', metavar='file.db', action=FileExists,
    help='')


def create_jobs(dbname, sleep=5, limit=None):
    njobs = 0
    with sqlite3.connect(dbname) as conn:
        while limit is None or njobs < limit:
            c = conn.cursor()
            c.execute('select * from runs where status = ? limit 1', (_PENDING,))
            res = c.fetchone()
            if res is not None:
                model, output, data, _, runid = res
                c.execute('update runs set status = ? where runid = ?', (_RUNNING, runid))
                conn.commit()
                njobs += 1
                yield model, output, data, runid
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
    # Theano flags
    pid = int(multiprocessing.current_process().name.split('-')[-1])
    gpu = (pid - 1) // 2
    env = os.environ.copy()
    env['THEANO_FLAGS'] = 'floatX=float32,warn_float64=warn,optimizer=fast_run,device=gpu{},scan.allow_gc=False,lib.cnmem=0.3'.format(gpu)

    # arglist
    model, output, data, runid = args
    arglist = ["/home/ubuntu/git/sloika/bin/train_network.py",
               "--lrdecay", "5000",
               "--window", "3",
               "--bad",
               model,
               output,
               data]

    proc = subprocess.Popen(arglist, env=env)
    proc.wait()
    return model, output, data, runid, proc.returncode


if __name__ == '__main__':
    args = parser.parse_args()

    jobs = create_jobs(args.database, sleep=args.sleep, limit=args.limit)
    pool = multiprocessing.Pool(8)
    for model, output, data, runid, returncode in imap_unordered(pool, run_job, jobs):
        status = _SUCCESS if returncode == 0 else _FAILURE
        with sqlite3.connect(args.database) as conn:
            c = conn.cursor()
            c.execute('update runs set status = ? where runid = ?', (status, runid))
    pool.close()
    pool.join()
