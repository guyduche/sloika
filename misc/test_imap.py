#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import next
from builtins import range
from builtins import *
from itertools import islice
import multiprocessing
import time

_SLEEP = 4
_NPROC = 4


def gen(n):
    for i in range(n):
        print('    Yielding', i)
        yield i


def worker(i):
    pid = int(multiprocessing.current_process().name.split('-')[-1])
    sleep_len = _SLEEP * (_NPROC - pid + 1)
    print('    Doing {}. Will sleep for {}s'.format(i, sleep_len))
    time.sleep(sleep_len)
    return i


def imap_unordered2(pool, f, iterable):
    from collections import deque
    q = deque()
    # would be better to use the number of processes
    # actually in the pool
    MAX = pool._processes
    for arg in iterable:
        while len(q) >= MAX:
            yield q.popleft().get()
        q.append(pool.apply_async(f, (arg,)))
    while q:
        yield q.popleft().get()


def imap_unordered3(pool, f, iterable):
    nproc = pool._processes

    #  Fill job array with at most nproc processes
    jobs = [pool.apply_async(f, (args,)) for args in islice(iterable, nproc)]

    #  Repeatedly scan through job array looking for new jobs
    while any(jobs):
        for i in range(len(jobs)):
            if jobs[i] is not None and jobs[i].ready():
                res = jobs[i].get()
                try:
                    next_job = next(iterable)
                    jobs[i] = pool.apply_async(f, (next_job,))
                except StopIteration:
                    jobs[i] = None

                yield res

if __name__ == '__main__':
    pool = multiprocessing.Pool(_NPROC)

    print('pool.imap_unordered')
    print('===================')
    print('Iterator is drained before jobs are started')
    print()
    t0 = time.time()
    gg = gen(12)
    for i in pool.imap_unordered(worker, gg):
        print('    Done', i)
    print('* Time ', time.time() - t0)

    print('http://bugs.python.org/issue19993')
    print('===================')
    print('Slowest job blocks, one extra is always taken from iterator')
    print()
    t0 = time.time()
    gg = gen(12)
    for i in imap_unordered2(pool, worker, gg):
        print('    Done', i)
    print('* Time ', time.time() - t0)

    print('Modified imap')
    print('=============')
    print()
    t0 = time.time()
    gg = gen(12)
    for i in imap_unordered3(pool, worker, gg):
        print('    Done', i)
    print('* Time ', time.time() - t0)
