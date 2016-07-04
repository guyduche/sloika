#!/usr/bin/env python
import argparse
import h5py
import numpy as np

from untangled.cmdargs import AutoBool, FileExists, Positive

parser = argparse.ArgumentParser(description='Sample from simple HDF5 file')
parser.add_argument('--size', default=None, type=Positive(int), help='Number of entries to sample')
parser.add_argument('--random', default=True, action=AutoBool, help='Randomise')
parser.add_argument('input', action=FileExists, help='Input HDF5 file')
parser.add_argument('output', help='Input HDF5 file')

if __name__ == '__main__':
    args = parser.parse_args()

    with h5py.File(args.input, 'r') as in5:
        seqs = list(in5.keys())
        if args.random:
            seqs = np.random.permutation(seqs)
        with h5py.File(args.output, 'w') as out5:
            for s in seqs[:args.size]:
                out5[s] = in5[s][:]
