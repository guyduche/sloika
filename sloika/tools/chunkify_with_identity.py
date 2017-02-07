from __future__ import print_function

import argparse
import h5py
import numpy as np
from scipy import linalg
import sys
import os

from sloika import batch
from sloika.config import sloika_dtype
from sloika.util import get_kwargs

from untangled.cmdargs import (AutoBool, FileAbsent, FileExists, Maybe,
                               NonNegative, Positive, proportion)
from untangled.iterators import imap_mp
from untangled import fast5


def progress_report(i):
    i += 1
    sys.stderr.write('.')
    if i % 50 == 0:
        print('{:8d}'.format(i))
    return i


def create_hdf5(args, all_chunks, all_labels, all_bad):
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(os.path.normpath(output_dir))

    #  Mark chunks with too many blanks with a zero weight
    nblank = np.sum(all_labels == 0, axis=1)
    max_blanks = int(all_labels.shape[1] * args.blanks)
    all_weights = nblank < max_blanks

    with h5py.File(args.output, 'w') as h5:
        bad_ds = h5.create_dataset('bad', all_bad.shape, dtype='i1',
                                   compression="gzip")
        chunk_ds = h5.create_dataset('chunks', all_chunks.shape, dtype='f4',
                                     compression="gzip")
        label_ds = h5.create_dataset('labels', all_labels.shape, dtype='i4',
                                     compression="gzip")
        weight_ds = h5.create_dataset('weights', all_weights.shape, dtype='f4',
                                      compression="gzip")
        bad_ds[:] = all_bad
        chunk_ds[:] = all_chunks
        label_ds[:] = all_labels
        weight_ds[:] = all_weights
        h5['/'].attrs['chunk'] = args.chunk_len
        h5['/'].attrs['kmer'] = args.kmer_len
        h5['/'].attrs['section'] = args.section
        h5['/'].attrs['trim'] = args.trim
        h5['/'].attrs['scaled'] = args.use_scaled

def chunkify_with_identity_main(argv, parser):
    parser.add_argument('input_folder', action=FileExists,
                        help='Directory containing single-read fast5 files')
    parser.add_argument('output', action=FileAbsent, help='Output HDF5 file')

    args = parser.parse_args(argv)

    fast5_files = fast5.iterate_fast5(args.input_folder, paths=True,
                                      limit=args.limit,
                                      strand_list=args.input_strand_list)

    print('* Processing data using', args.threads, 'threads')

    kwarg_names = ['section', 'chunk_len', 'kmer_len', 'min_length', 'trim', 'use_scaled', 'normalise']
    i = 0
    bad_list = []
    chunk_list = []
    label_list = []
    for chunks, labels, bad_ev in imap_mp(batch.chunk_worker, fast5_files, threads=args.threads,
                                          unordered=True, fix_kwargs=get_kwargs(args, kwarg_names)):
        if chunks is not None and labels is not None:
            i = progress_report(i)
            chunk_list.append(chunks)
            label_list.append(labels)
            bad_list.append(bad_ev)

    all_chunks = np.vstack(chunk_list)
    all_labels = np.vstack(label_list)
    all_bad = np.vstack(bad_list)

    print('\n* Writing out to HDF5')
    create_hdf5(args, all_chunks, all_labels, all_bad)
