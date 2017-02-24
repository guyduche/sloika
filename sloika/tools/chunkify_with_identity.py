from __future__ import print_function

import argparse
import os
import sys
import numpy as np

from sloika import batch, util

from untangled.cmdargs import (AutoBool, FileAbsent, FileExists, Maybe,
                               NonNegative, Positive, proportion)
from untangled.iterators import imap_mp
from untangled import fast5


def chunkify_with_identity_main(argv, parser):
    args = parser.parse_args(argv)

    if not args.overwrite:
        if os.path.exists(args.output):
            print("Cowardly refusing to overwrite {}".format(args.output))
            sys.exit(1)

    fast5_files = fast5.iterate_fast5(args.input_folder, paths=True,
                                      limit=args.limit,
                                      strand_list=args.input_strand_list)

    print('* Processing data using', args.jobs, 'threads')

    kwarg_names = ['section', 'chunk_len', 'kmer_len', 'min_length', 'trim', 'use_scaled', 'normalisation']
    i = 0
    bad_list = []
    chunk_list = []
    label_list = []
    for chunks, labels, bad_ev in imap_mp(batch.chunk_worker, fast5_files, threads=args.jobs,
                                          unordered=True, fix_kwargs=util.get_kwargs(args, kwarg_names)):
        if chunks is not None and labels is not None:
            i = util.progress_report(i)
            chunk_list.append(chunks)
            label_list.append(labels)
            bad_list.append(bad_ev)

    print('\n* Writing out to HDF5')
    util.create_hdf5(args, chunk_list, label_list, bad_list)
