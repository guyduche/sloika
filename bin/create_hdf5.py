#!/usr/bin/env python
import argparse
import h5py

from sloika import batch
from sloika.features import NFEATURES

from untangled.cmdargs import (AutoBool, FileExists, Maybe, Positive)
from untangled import fast5

parser = argparse.ArgumentParser(
    description = 'Create HDF file of a dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--bad', default=True, action=AutoBool,
    help='Label bad emissions')
parser.add_argument('--chunk', default=500, metavar='events', type=Positive(int),
    help='Length of each read chunk')
parser.add_argument('--kmer', default=5, metavar='length', type=Positive(int),
    help='Length of kmer to estimate')
parser.add_argument('--limit', default=None, type=Maybe(Positive(int)),
    help='Limit number of reads to process.')
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
parser.add_argument('--window', default=3, type=Positive(int), metavar='length',
    help='Window length for input features')
parser.add_argument('input_folder', action=FileExists,
    help='Directory containing single-read fast5 files.')
parser.add_argument('output', help='Output HDF5 file')

if __name__ == '__main__':
    args = parser.parse_args()


    fast5_files = set(fast5.iterate_fast5(args.input_folder, paths=True,
                                          limit=args.limit,
                                          strand_list=args.strand_list))


    all_chunks = None
    all_labels = None
    for chunks, labels in batch.kmers(fast5_files, args.section, None,
                                      args.chunk, args.window, args.kmer,
                                      trim=args.trim, bad=args.bad,
                                      use_scaled=args.use_scaled):
	all_chunks = np.vstack((all_chunks, chunks)) if all_chunks is not None else chunks
        all_labels = np.vstack((all_labels, labels)) if all_labels is not None else labels
        
   
    Rotation = np.identity(all_chunks.shape[-1])
    if args.orthogonal:
        chunk_shape = all_chunks.shape
        all_chunks = all_chunks.reshape(-1, chunk_shape[-1])
        V = linalg.blas.ssyrk(1.0, full_chunks, trans=True) / np.float32(len(full_chunks))
        w, E = linalg.eigh(V)
        Rotation = E / np.sqrt(w.reshape(1, -1))
        all_chunks = linalg.blas.sgemm(1.0, all_chunks, Rotation, trans_b=True)
        all_chunks = all_chunks.reshape(chunk_shape)

    with h5py.File(args.ouput, 'w') as h5:
        h5['chunks'] = all_chunks
        h5['labels'] = all_labels 
        h5['rotation'] = Rotation
