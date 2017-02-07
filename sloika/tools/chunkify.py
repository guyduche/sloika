from __future__ import print_function

import argparse
import h5py
import numpy as np
from scipy import linalg
import sys
import os

from sloika import batch
from sloika.config import sloika_dtype

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
    print('\n* Writing out to HDF5')

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
        h5['/'].attrs['chunk'] = args.chunk
        h5['/'].attrs['kmer'] = args.kmer
        h5['/'].attrs['section'] = args.section
        h5['/'].attrs['trim'] = args.trim
        h5['/'].attrs['scaled'] = args.use_scaled


def chunkify_main(argv):
    program_name = ' '.join(sys.argv[:2])

    parser = argparse.ArgumentParser(prog=program_name,
                                     description='Create HDF file of a dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--blanks', metavar='proportion', default=0.7,
                        type=proportion, help='Maximum proportion of blanks in labels')
    parser.add_argument('--chunk', default=500, metavar='events', type=Positive(int),
                        help='Length of each read chunk')
    parser.add_argument('--kmer', default=5, metavar='length', type=Positive(int),
                        help='Length of kmer to estimate')
    parser.add_argument('--threads', default=8, metavar='n', type=Positive(int),
                        help='Number of threads to use when processing data')
    parser.add_argument('--limit', default=None, type=Maybe(Positive(int)),
                        help='Limit number of reads to process.')
    parser.add_argument('--min_length', default=1200, metavar='events',
                        type=Positive(int), help='Minimum events in acceptable read')
    parser.add_argument('--normalise', default=True, action=AutoBool,
                        help='Per-strand normalisation')
    parser.add_argument('--section', default='template',
                        choices=['template', 'complement'], help='Section to call')
    parser.add_argument('--strand_list', default=None, action=FileExists,
                        help='strand summary file containing subset.')
    parser.add_argument('--trim', default=(50, 10), nargs=2, type=NonNegative(int),
                        metavar=('beginning', 'end'),
                        help='Number of events to trim off start and end')
    parser.add_argument('--use_scaled', default=False, action=AutoBool,
                        help='Train from scaled event statistics')
    parser.add_argument('input_folder', action=FileExists,
                        help='Directory containing single-read fast5 files.')
    parser.add_argument('output', action=FileAbsent, help='Output HDF5 file')

    args = parser.parse_args(argv[1:])

    fast5_files = fast5.iterate_fast5(args.input_folder, paths=True,
                                      limit=args.limit,
                                      strand_list=args.strand_list)

    print('* Processing data using', args.threads, 'threads')
    fix_kwargs = {'section': args.section,
                  'chunk_len': args.chunk,
                  'kmer_len': args.kmer,
                  'min_length': args.min_length,
                  'trim': args.trim,
                  'use_scaled': args.use_scaled,
                  'normalise': args.normalise,
                  }
    i = 0
    bad_list = []
    chunk_list = []
    label_list = []
    for chunks, labels, bad_ev in imap_mp(batch.chunk_worker, fast5_files, threads=args.threads,
                                          fix_kwargs=fix_kwargs):
        if chunks is not None and labels is not None:
            i = progress_report(i)
            chunk_list.append(chunks)
            label_list.append(labels)
            bad_list.append(bad_ev)

    all_chunks = np.vstack(chunk_list)
    all_labels = np.vstack(label_list)
    all_bad = np.vstack(bad_list)

    create_hdf5(args, all_chunks, all_labels, all_bad)
