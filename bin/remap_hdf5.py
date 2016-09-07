#!/usr/bin/env python 
import argparse
import h5py
import numpy as np
import numpy.lib.recfunctions as nprf
import sys
import time

from untangled import bio, fast5
from untangled.cmdargs import (AutoBool, FileExists, NonNegative, Positive, Maybe)
from untangled.iterators import imap_mp


# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Map transducer to reference sequence',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--jobs', default=8, type=Positive(int),
    help='Number of jobs to run in parallel')
parser.add_argument('--slip', default=None, metavar='score', type=Maybe(NonNegative(float)),
    help='Slip penalty')
parser.add_argument('model', action=FileExists, help='Pickled model file')
parser.add_argument('input', help='HDF5 file for input')
parser.add_argument('output', help='HDF5 for output')

def compress_labels(labels, klen):
    """ Make sequece from labels

    :param labels: A :class:`ndarray` (batch x chunk) containing labels
    :param klen: Length of kmer

    :returns: A tuple of a :class:`ndarray` containing the concatenated
    references sequences and a :class:`ndarrray` containing the reference
    lengths
    """
    state_to_kmer = bio.all_kmers(klen)
    kmer_to_state = bio.kmer_mapping(klen)

    label_kmers = [state_to_kmer[s - 1] for s in labels if s > 0]
    seq = bio.kmers_to_sequence(label_kmers, homopolymer_step=True)
    seq_kmers = bio.seq_to_kmers(seq, klen)
    label_vec = np.array([kmer_to_state[k] for k in seq_kmers])

    # Adjust label states so 0 is blank
    label_vec += 1
    return label_vec.astype(np.int32)


def map_transducer(args, idx):
    import cPickle
    from sloika import transducer
    try:
        with h5py.File(args.input, 'r') as h5:
            inMat = h5['chunks'][idx]
            lbls = h5['labels'][idx]
        inMat = np.expand_dims(inMat, axis=1)
    except:
        return None

    with open(args.model, 'r') as fh:
        calc_post = cPickle.load(fh)

    post = np.squeeze(calc_post(inMat))
    seq = compress_labels(lbls, args.kmer)

    score, path = transducer.map_to_sequence(post, seq, slip=args.slip, log=False)
    labels = seq[path]
    labels[np.ediff1d(path, to_begin=1) == 0] = 0

    return score, np.squeeze(inMat), labels, path


if __name__ == '__main__':
    args = parser.parse_args()

    with h5py.File(args.output, 'w') as h5:
        with h5py.File(args.input, 'r') as h5in:
            bad_ds = h5.create_dataset('bad', h5in['bad'].shape, dtype='i1')
            chunk_ds = h5.create_dataset('chunks', h5in['chunks'].shape, dtype='f4')
            label_ds = h5.create_dataset('labels', h5in['labels'].shape, dtype='i4')
            path_ds = h5.create_dataset('paths', h5in['labels'].shape, dtype='i4')

            h5['rotation'] = h5in['rotation'][()]
            for k, v in h5in.attrs.items():
                h5.attrs[k] = v

            args.kmer = h5in.attrs['kmer']
            nchunk, chunk_len = label_ds.shape


        for idx, res in enumerate(imap_mp(map_transducer, xrange(nchunk), threads=args.jobs, fix_args=[args], unordered=False)):
            if res is None:
                continue
            score, inMat, labels, path = res
            chunk_ds[idx] = inMat
            label_ds[idx] = labels
            path_ds[idx] = path
            sys.stderr.write('.')
            if (idx + 1) % 50 == 0:
                sys.stderr.write('{:6d}\n'.format((idx + 1) // 50))
        sys.stderr.write('\n')
