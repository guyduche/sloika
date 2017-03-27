from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import os
import sys
import h5py
import numpy as np


def geometric_prior(n, m, rev=False):
    """ Make a vec

    :param n: length of vector
    :param m: mean of distribution.
    :param rev: Reverse distribution

    :returns: A 1D :class:`ndarray` containing log probabilities
    """
    p = 1.0 / (1.0 + m)
    prior = np.repeat(np.log(p), n)
    prior[1:] += np.arange(1, n) * np.log1p(-p)
    if rev:
        prior = prior[::-1]
    return prior


def is_contiguous(ndarray):
    '''
    See https://docs.scipy.org/doc/numpy/reference/generated/numpy.ascontiguousarray.html
    '''
    return ndarray.flags['C_CONTIGUOUS']


def get_kwargs(args, names):
    kwargs = {}
    for name in names:
        kwargs[name] = getattr(args, name)
    return kwargs


def progress_report(i):
    i += 1
    sys.stderr.write('.')
    if i % 50 == 0:
        print('{:8d}'.format(i))
    return i


def create_hdf5(output, blanks, attributes, chunk_list, label_list, bad_list):
    assert len(chunk_list) == len(label_list) == len(bad_list)
    assert len(chunk_list) > 0

    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(os.path.normpath(output_dir))

    all_chunks = np.concatenate(chunk_list)
    all_labels = np.concatenate(label_list)
    all_bad = np.concatenate(bad_list)

    #  Mark chunks with too many blanks with a zero weight
    nblank = np.sum(all_labels == 0, axis=1)
    max_blanks = int(all_labels.shape[1] * blanks)
    all_weights = nblank < max_blanks

    with h5py.File(output, 'w') as h5:
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

        for (key, value) in attributes.items():
            h5['/'].attrs[key] = value


def trim_array(x, from_start, from_end):
    from_end = None if from_end == 0 else -from_end
    return x[from_start:from_end]
