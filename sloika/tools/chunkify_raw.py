from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

import numpy as np
from sloika import features
from untangled import bio, fast5
from untangled.iterators import imap_mp
from untangled.maths import med_mad, studentise, mad


def fast_mode(X):
    """Apply mode over the last axis of an array

    This function contains explicit calculations for applying the mode over
    the final axis of an array of shape [..., k] for k = 1, 2, 3, 4. For k > 4,
    we use scipy's method (scipy.stats.mode).

    In addition, this function is stable for k < 5 in the sense that the first
    modal value is always selected, i.e. [1, 4, 4, 1] --> 1. The scipy method
    does not enforce this property.

    This method is a helper function for downsampling labelled time-series
    using majority labels. If the downsampling factor is small, this method is
    faster and less memory-intensive than scipy.stats.mode.
    """
    k = X.shape[-1]

    if k == 0:
        return X
    elif k == 1 or k == 2 or X.size == 0:
        return X[..., 0]
    elif k == 3:
        out = np.zeros(X.shape[:-1], X.dtype)
        # don't be a RAM hog, use a loop
        nbatch = int(np.ceil(X.size / 5e9))
        batch = max(1, X.shape[0] // nbatch)
        for i in range(0, X.shape[0], batch):
            Z = X[i:i + batch]
            out[i:i + batch] = np.where(Z[..., 1] == Z[..., 2],
                                        Z[..., 1], Z[..., 0])
        return out
    elif k == 4:
        out = np.zeros(X.shape[:-1], X.dtype)
        # don't be a RAM hog, use a loop
        nbatch = int(np.ceil(X.size / 5e9))
        batch = max(1, X.shape[0] // nbatch)
        for i in range(0, X.shape[0], batch):
            Z = X[i:i + batch]
            out[i:i + batch] = np.where(Z[..., 0] == Z[..., 1], Z[..., 0],
                                    np.where(Z[..., 0] == Z[..., 2], Z[..., 0],
                                    np.where(Z[..., 0] == Z[..., 3], Z[..., 0],
                                    np.where(Z[..., 1] == Z[..., 2], Z[..., 1],
                                    np.where(Z[..., 1] == Z[..., 3], Z[..., 1],
                                    np.where(Z[..., 2] == Z[..., 3], Z[..., 2],
                                    Z[..., 0]))))))
        return out
    else:
        return scipy.stats.mode(X, len(X.shape) - 1).mode[:,:,0]


def majority_labels(labels, k):
    """Downsample labels along final axis using majority voting"""
    if k == 1:
        # "Here I am, brain the size of a planet, and they ask me to
        # downsample by a factor of 1"
        return labels

    shape = labels.shape[:-1]
    t = labels.shape[-1]
    d = len(labels.shape)

    if t % k:
        pad = np.zeros(shape + (k - t % k,), labels.dtype)
        labels = np.concatenate([labels, pad], d - 1)
        t = labels.shape[-1]

    N, T = np.indices(labels.shape)

    starts = np.where(labels, T, 0)
    starts = np.maximum.accumulate(starts, 1)
    starts = starts.reshape(shape + (t / k, k))
    starts = fast_mode(starts)

    lab = labels[N[:,::k], starts]
    return (lab * (np.apply_along_axis(np.ediff1d, 1, starts, to_begin=1) > 0)).astype('i4')


def interpolate_pos(ev, att):
    """Return a function: time -> reference position by interpolating mapping

    :param ev: mapping table with fields start, length, seq_pos and kmer
    :param att: mapping attributes direction, ref_start, ref_stop
        (ev, att) could be returned by f5file.get_any_mapping_data()
    """
    def interp(t, k=5):
        EPS = 10**-10 # small value for avoiding round to even

        ev_mid = ev['start'] + 0.5 * ev['length']
        map_k = len(ev['kmer'][0])

        if att['direction'] == "+":
            map_ref_pos = ev['seq_pos'] + 0.5 * map_k - att['ref_start']
            pos_interp = np.interp(t, ev_mid, map_ref_pos)
            pos = np.around(pos_interp - 0.5 * k + EPS).astype(int)
            return pos
        else:
            map_ref_pos = att['ref_stop'] - ev['seq_pos'] + 0.5 * map_k
            pos_interp = np.around(np.interp(t, ev_mid, map_ref_pos))
            pos = np.around(pos_interp - 0.5 * k + EPS).astype(int)
            return pos

    return interp


def interpolate_labels(ev, att):
    """Return a function: time -> reference kmer by interpolating mapping

    :param ev: mapping table with fields start, length, seq_pos and kmer
    :param att: mapping attributes reference, direction, ref_start, ref_stop
        (ev, att) could be returned by f5file.get_any_mapping_data()
    """
    def interp(t, k=5):
        pos = interpolate_pos(ev, att)(t, k)
        return [att['reference'][i: i + k] for i in pos]

    return interp


def kmers_to_labels(kmer_array, kmer_len, index_from=1):
    """Extract shortened kmers from an array of kmers

    :param kmer_array: a numpy array of kmers
    :param kmer_len: length of sequence context used to determine label

    :returns: an array of labels
    """
    kmer_array = np.ascontiguousarray(kmer_array)

    old_kmer_len = len(kmer_array.flat[0])
    assert kmer_len <= old_kmer_len

    offset = (old_kmer_len - kmer_len + 1) // 2
    extracted = np.chararray(kmer_array.shape, kmer_len,
            buffer=kmer_array.data, offset=offset, strides=kmer_array.strides)
    mapping = bio.kmer_mapping(kmer_len)
    labels = np.array(map(lambda k: mapping[k], kmer_array.flat)) + index_from

    return labels.reshape(kmer_array.shape)


def remove_same(arr):
    """Replace repeated elements in 1d array with 0"""
    arr[np.ediff1d(arr, to_begin=1) == 0] = 0
    return arr


def fill_zeros_with_prev(arr):
    """Fills zero values with previous value"""
    ix = np.add.accumulate(arr != 0).astype(int) - 1
    return arr[arr != 0][ix]


def raw_chunkify_worker(fn, section, chunk_len, kmer_len, min_length, trim, normalise,
                downsample_factor, downsample_method="interpolation"):
    """  Worker for creating labelled features from raw data

    Although this is a raw data method, we assume the existence in the fast5
    file of a mapping table of events (or other signal segmentation) with
    fields 'start' and 'kmer'

    :param fn: A filename to read from.
    :param section: Section of read to process (template / complement)
    :param chunk_len: Length on each chunk
    :param kmer_len: Kmer length for training
    :param min_length: Minumum number of samples before read can be considered.
    :param trim: Tuple (beginning, end) of number of samples to trim from read.
    :param normalise: Do per-strand normalisation
    :param downsample_factor: factor by which to downsample labels
    :param downsample_method: method to use for downsampling, either
        "simple", "majority" or "interpolation"
    """
    kmer_to_state = bio.kmer_mapping(kmer_len)

    try:
        with fast5.Reader(fn) as f5:
            ev, att = f5.get_any_mapping_data(section)
            sig = f5.get_read(raw=True)
            sample_rate = f5.sample_rate
            start_sample = int(f5['Raw/Reads'].values()[0].attrs['start_time'])
    except:
        return fn, None, None, None

    ev['move'][0] = 1

    # start_sample is a uint64. When you do arithmetic with it,
    # numpy tends to return a float
    begin, end = trim
    map_start = int(round(ev['start'][0] * sample_rate)
                    - start_sample + begin)
    map_end = int(round((ev['start'][-1] + ev['length'][-1]) * sample_rate)
                    - start_sample - end)
    sig_trim = sig[map_start:map_end]

    if len(sig_trim) < min(chunk_len, min_length):
        return fn, None, None, None

    if normalise:
        loc, scale = med_mad(sig_trim)
        sig_trim = (sig_trim - loc) / scale

    #  Create feature array
    ml = len(sig_trim) // chunk_len
    sig_trim = sig_trim[:ml * chunk_len].reshape((ml, chunk_len, 1))

    if downsample_method in ["simple", "majority"]:
        #  Create label array
        model_kmer_len = len(ev['kmer'][0])
        ub = chunk_len * ml
        # Use rightmost middle kmer
        kl = (model_kmer_len - kmer_len + 1) // 2
        ku = kl + kmer_len
        new_labels = 1 + np.array(map(lambda k: kmer_to_state[k[kl : ku]],
                                      ev['kmer'][ev['move'] > 0]), dtype=np.int32)
        new_labels = np.concatenate([[0,], new_labels])

        label_start = (np.around(ev['start'][ev['move'] > 0] * sample_rate).astype(int)
                                      - start_sample - map_start)
        label_start = label_start[label_start < ub].data

        if downsample_method == "simple":
            idx = np.zeros(ub, dtype=int)
            idx[label_start] = np.arange(label_start.shape[0]) + 1
            idx = fill_zeros_with_prev(idx)
            idx = idx.reshape((ml, chunk_len))[:,::downsample_factor]
            idx = np.apply_along_axis(remove_same, 1, idx)

            sig_labels = new_labels[idx]
        else:
            idx = np.zeros(ub, dtype=int)
            idx[label_start] = np.arange(label_start.shape[0]) + 1
            idx = fill_zeros_with_prev(idx)
            idx = idx.reshape((ml, chunk_len))
            idx = np.apply_along_axis(remove_same, 1, idx)

            sig_labels = new_labels[idx]
            sig_labels = majority_labels(sig_labels, downsample_factor)
    elif downsample_method == "interpolation":
        delta = 1.0 / sample_rate
        chunk_delta = delta * chunk_len
        t0 = (map_start + start_sample) / sample_rate
        tch = np.arange(t0, t0 + chunk_delta, downsample_factor * delta)
        t = np.array([tch + i * chunk_delta for i in range(ml)]).flatten()
        pos = interpolate_pos(ev, att)(t, kmer_len)
        kmers = interpolate_labels(ev, att)(t, kmer_len)
        sig_labels = 1 + np.array(map(lambda k: kmer_to_state[k], kmers),
                                  dtype=np.int32)
        sig_labels[np.ediff1d(pos, to_begin=1) == 0] = 0
        sig_labels = sig_labels.reshape((ml, -1))
    else:
        raise ValueError("downsample_method not understood")

    sig_bad = np.zeros((ml, chunk_len), dtype=bool)

    return (np.ascontiguousarray(sig_trim),
            np.ascontiguousarray(sig_labels),
            np.ascontiguousarray(sig_bad))
