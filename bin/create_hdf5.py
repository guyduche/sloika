#!/usr/bin/env python
import argparse
import h5py
import numpy as np
from scipy import linalg
import sys

from sloika import batch, sloika_dtype
from sloika.features import NFEATURES

from untangled.cmdargs import (AutoBool, FileExists, Maybe, Positive)
from untangled import fast5

parser = argparse.ArgumentParser(
    description = 'Create HDF file of a dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--chunk', default=500, metavar='events', type=Positive(int),
    help='Length of each read chunk')
parser.add_argument('--kmer', default=5, metavar='length', type=Positive(int),
    help='Length of kmer to estimate')
parser.add_argument('--limit', default=None, type=Maybe(Positive(int)),
    help='Limit number of reads to process.')
parser.add_argument('--normalise', default=True, action=AutoBool,
    help='Per-strand normalisation')
parser.add_argument('--orthogonal', default=False, action=AutoBool,
    help='Make input features orthogonal')
parser.add_argument('--section', default='template',
    choices=['template', 'complement'], help='Section to call')
parser.add_argument('--strand_list', default=None, action=FileExists,
    help='strand summary file containing subset.')
parser.add_argument('--trim', default=(500, 50), nargs=2, type=Positive(int),
    metavar=('beginning', 'end'),
    help='Number of events to trim off start and end')
parser.add_argument('--use_scaled', default=False, action=AutoBool,
    help='Train from scaled event statistics')
parser.add_argument('input_folder', action=FileExists,
    help='Directory containing single-read fast5 files.')
parser.add_argument('output', help='Output HDF5 file')

if __name__ == '__main__':
    args = parser.parse_args()


    fast5_files = set(fast5.iterate_fast5(args.input_folder, paths=True,
                                          limit=args.limit,
                                          strand_list=args.strand_list))

    bad_list = []
    chunk_list = []
    label_list = []
    print '* Reading in data'
    for i, (chunks, labels, bad) in enumerate(batch.kmers(fast5_files, args.section,
                                                          args.chunk, args.kmer,
                                                          trim=args.trim,
                                                          use_scaled=args.use_scaled,
                                                          normalise=args.normalise)):
        sys.stderr.write('.')
        if (i + 1) % 50 == 0:
            print '{:8d}'.format(i + 1)
        chunk_list.append(chunks)
        label_list.append(labels)
        bad_list.append(bad)

    nchunks = sum(map(lambda x: len(x), label_list))
    nfeature = chunk_list[0].shape[-1]
    label_len = label_list[0].shape[-1]
    all_chunks = np.empty((nchunks, args.chunk, nfeature), dtype=sloika_dtype)
    all_labels = np.empty((nchunks, label_len), dtype=np.int32)
    all_bad = np.empty((nchunks, label_len), dtype=np.int8)
    idx = 0
    for chunk in chunk_list:
        chunk_size = len(chunk)
        all_chunks[idx : idx + chunk_size] = chunk
        idx += chunk_size
    idx = 0
    for label in label_list:
        label_size = len(label)
        all_labels[idx : idx + label_size] = label
        idx += label_size
    idx = 0
    for bad in bad_list:
        bad_size = len(bad)
        all_bad[idx : idx + bad_size] = bad
        idx += bad_size


    rotation = np.identity(all_chunks.shape[-1])
    centre = np.zeros(all_chunks.shape[-1])
    if args.orthogonal:
        print '\n* Doing orthogonalisation'
        chunk_shape = all_chunks.shape
        all_chunks = all_chunks.reshape(-1, chunk_shape[-1])
        # Centre
        centre = np.mean(all_chunks, axis=0, dtype=np.float64).astype(sloika_dtype)
        all_chunks -= centre
        
        # Rotate 
        V = linalg.blas.ssyrk(1.0, all_chunks, trans=True, lower=True) / np.float32(len(all_chunks))
        V = V + V.T - np.diag(np.diag(V))
        L0 = linalg.cho_factor(V, lower=True)

        all_chunks = linalg.solve_triangular(L0[0], all_chunks.T, trans=False, lower=L0[1])
        all_chunks = np.ascontiguousarray(all_chunks.T)
        
        all_chunks = all_chunks.reshape(chunk_shape)

    print '* Writing out to HDF5'
    with h5py.File(args.output, 'w') as h5:
        bad_ds = h5.create_dataset('bad', all_bad.shape, dtype='i1')
        chunk_ds = h5.create_dataset('chunks', all_chunks.shape, dtype='f4')
        label_ds = h5.create_dataset('labels', all_labels.shape, dtype='i4')
        bad_ds[:] = all_bad
        chunk_ds[:] = all_chunks
        label_ds[:] = all_labels
        h5['rotation'] = rotation
        h5['centre'] = centre
        h5['/'].attrs['chunk'] = args.chunk
        h5['/'].attrs['kmer'] = args.kmer
        h5['/'].attrs['section'] = args.section
        h5['/'].attrs['trim'] = args.trim
        h5['/'].attrs['scaled'] = args.use_scaled
