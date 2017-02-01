#!/usr/bin/env python
import argparse
import cPickle
import h5py
import numpy as np
import numpy.lib.recfunctions as nprf
import sys
import time

from untangled import bio, fast5
from untangled.cmdargs import (AutoBool, display_version_and_exit, FileAbsent,
                               FileExists, NonNegative, Positive, Maybe)
from untangled.iterators import imap_mp, izip

from sloika import features, helpers, transducer, __version__


# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Map transducer to reference sequence',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch', metavar='size', default=1000, type=Positive(int),
    help='Number of posterior matrices to calculate simulataneously on GPU')
parser.add_argument('--compile', default=None, action=FileAbsent,
    help='File output compiled model')
parser.add_argument('--jobs', default=8, type=Positive(int),
    help='Number of jobs to run in parallel')
parser.add_argument('--slip', default=None, metavar='penalty', type=Maybe(NonNegative(float)),
    help='Slip penalty')
parser.add_argument('--version', nargs=0, action=display_version_and_exit, metavar=__version__,
    help='Display version information.')
parser.add_argument('model', action=FileExists, help='Pickled model file')
parser.add_argument('input', action=FileExists, help='HDF5 file for input')
parser.add_argument('output', action=FileAbsent, help='HDF5 for output')

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
    seq = bio.kmers_to_sequence(label_kmers, always_move=True)
    seq_kmers = bio.seq_to_kmers(seq, klen)
    label_vec = np.array([kmer_to_state[k] for k in seq_kmers])

    # Adjust label states so 0 is blank
    label_vec += 1
    return label_vec.astype(np.int32)


def map_transducer(args, fargs):
    idx, inmat, post, lbls = fargs
    try:
        seq = compress_labels(lbls, args.kmer)

        score, path = transducer.map_to_sequence(post, seq, slip=args.slip, log=False)
        labels = seq[path]
        labels[np.ediff1d(path, to_begin=1) == 0] = 0
    except:
        return idx, inmat, lbls, np.zeros(len(inmat))

    return idx, inmat, labels, path


if __name__ == '__main__':
    args = parser.parse_args()

    #  Compile model file if necessary and read in compiled model
    compiled_file = helpers.compile_model(args.model, args.compile)
    with open(compiled_file, 'r') as fh:
        calc_post = cPickle.load(fh)

    with h5py.File(args.output, 'w') as h5:
        with h5py.File(args.input, 'r') as h5in:
            bad_ds = h5.create_dataset('bad', h5in['bad'].shape, dtype='i1',
                                       compression="gzip")
            chunk_ds = h5.create_dataset('chunks', h5in['chunks'].shape,
                                         dtype='f4', compression="gzip")
            label_ds = h5.create_dataset('labels', h5in['labels'].shape,
                                         dtype='i4', compression="gzip")
            path_ds = h5.create_dataset('paths', h5in['labels'].shape,
                                        dtype='i4', compression="gzip")

            h5['weights'] = h5in['weights'][()]

            for k, v in h5in.attrs.items():
                h5.attrs[k] = v

            args.kmer = h5in.attrs['kmer']
            nchunk, chunk_len = label_ds.shape

            nbatch = nchunk // args.batch
            ichunk = 0
            for bidx in xrange(nbatch+1):
                idx = bidx * args.batch
                inMat = h5in['chunks'][idx : idx + args.batch]
                inMat = inMat.transpose(1, 0, 2)
                lbls = h5in['labels'][idx : idx + args.batch]

                post = calc_post(inMat).transpose(1, 0, 2)
                inMat = inMat.transpose(1, 0, 2)
                assert len(lbls) == len(inMat)
                assert len(lbls) == len(post)
                idata = izip(xrange(idx, idx + len(lbls)), inMat, post, lbls)

                for res in imap_mp(map_transducer, idata, threads=args.jobs, fix_args=[args], unordered=True):
                    idx, inMat, labels, path = res
                    chunk_ds[idx] = inMat
                    label_ds[idx] = labels
                    path_ds[idx] = path
                    sys.stderr.write('.')
                    ichunk += 1
                    if ichunk % 50 == 0:
                        sys.stderr.write('{:6d} {:8d}\n'.format(ichunk // 50, ichunk))

    sys.stderr.write('\n')
