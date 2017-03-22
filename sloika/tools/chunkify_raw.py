from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

import numpy as np
import os

from sloika import util
from untangled import bio, fast5
from untangled.iterators import imap_mp
from untangled.maths import med_mad, studentise, mad

from untangled.cmdargs import (AutoBool, FileAbsent, FileExists, Maybe,
                               NonNegative, Positive, proportion)


DEFAULT_NORMALISATION = 'per-read'

AVAILABLE_NORMALISATIONS = frozenset(['none', 'per-read', 'per-chunk'])


def commensurate_mapping_to_raw(mapping_table, start_sample, sample_rate):
    """Replace time coordinates in mapping_table with indices into raw signal

    :param mapping_table: array of events (or similar) for a mapped read with
        start times and and lengths measured in seconds
    :param start_sample: start sample of the read raw signal
    :param sample_rate: number of samples per second

    :returns: mapping table with start times measured in samples from the start
        of the raw signal, and lengths measured in samples
    """
    new_field_types = {'start': '<i8', 'length': '<i8'}

    old_dtype = mapping_table.dtype.descr
    new_dtype = list(map(lambda (name, dtype): (name, new_field_types.get(name, dtype)), old_dtype))

    starts = np.around(mapping_table['start'] * sample_rate - start_sample).astype(int)
    lengths = np.around(mapping_table['length'] * sample_rate).astype(int)

    new_mapping_table = mapping_table.astype(new_dtype)
    new_mapping_table['start'] = starts
    new_mapping_table['length'] = lengths

    return new_mapping_table


def interpolate_pos(mapping_table, att):
    """Return a function: time -> reference position by interpolating mapping

    :param mapping_table: mapping table with fields start, length, seq_pos and kmer
    :param att: mapping attributes direction, ref_start, ref_stop
        (mapping_table, att) could be returned by f5file.get_any_mapping_data()
    """
    def interp(t, k=5):
        EPS = 10**-10 # small value for avoiding round to even

        ev_mid = mapping_table['start'] + 0.5 * mapping_table['length']
        map_k = len(mapping_table['kmer'][0])

        if att['direction'] == "+":
            map_ref_pos = mapping_table['seq_pos'] + 0.5 * map_k - att['ref_start']
            pos_interp = np.interp(t, ev_mid, map_ref_pos)
            pos = np.around(pos_interp - 0.5 * k + EPS).astype(int)
            return pos
        else:
            map_ref_pos = att['ref_stop'] - mapping_table['seq_pos'] + 0.5 * map_k
            pos_interp = np.around(np.interp(t, ev_mid, map_ref_pos))
            pos = np.around(pos_interp - 0.5 * k + EPS).astype(int)
            return pos

    return interp


def interpolate_labels(mapping_table, att):
    """Return a function: time -> reference kmer by interpolating mapping

    :param mapping_table: mapping table with fields start, length, seq_pos and kmer
    :param att: mapping attributes reference, direction, ref_start, ref_stop
        (mapping_table, att) could be returned by f5file.get_any_mapping_data()
    """
    def interp(t, k=5):
        pos = interpolate_pos(mapping_table, att)(t, k)
        return [att['reference'][i: i + k] for i in pos]

    return interp


def labels_from_mapping_table(kmer_array, kmer_len, index_from=1):
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
    labels = np.array(list(map(lambda k: mapping[k], extracted.flat))) + index_from

    return labels.reshape(kmer_array.shape).astype('i4')


def remove_same(arr):
    """Replace repeated elements in 1d array with 0"""
    arr[np.ediff1d(arr, to_begin=1) == 0] = 0
    return arr


def fill_zeros_with_prev(arr):
    """Fills zero values with previous value"""
    ix = np.add.accumulate(arr != 0).astype(np.int) - 1
    return arr[arr != 0][ix]


def raw_chunkify(signal, mapping_table, chunk_len, kmer_len, normalisation, downsample_factor, interpolation):
    assert len(signal) >= chunk_len
    assert normalisation in AVAILABLE_NORMALISATIONS

    ml = len(signal) // chunk_len
    new_inMat = signal[:ml * chunk_len].reshape((ml, chunk_len, 1))

    if normalisation == "per-chunk":
        chunk_medians = np.median(new_inMat, axis=1, keepdims=True)
        chunk_mads = mad(new_inMat, axis=1, keepdims=True)
        new_inMat = (new_inMat - chunk_medians) / chunk_mads
    elif normalisation == "per-read":
        new_inMat = (new_inMat - np.median(new_inMat)) / mad(new_inMat)
    else:
        assert normalisation == "none"

    if interpolation:
        t = np.arange(ml * chunk_len)
        pos = interpolate_pos(mapping_table, att)(t, kmer_len)
        kmers = interpolate_labels(mapping_table, att)(t, kmer_len)
        sig_labels = 1 + np.array(map(lambda k: kmer_to_state[k], kmers),
                                  dtype=np.int32)
        sig_labels[np.ediff1d(pos, to_begin=1) == 0] = 0
        sig_labels = sig_labels.reshape((ml, chunk_len))
    else:
        #  Create label array
        model_kmer_len = len(mapping_table['kmer'][0])
        ub = chunk_len * ml
        # Use rightmost middle kmer
        kl = (model_kmer_len - kmer_len + 1) // 2
        ku = kl + kmer_len
        new_labels = labels_from_mapping_table(mapping_table['kmer'][mapping_table['move'] > 0], kmer_len)
        new_labels = np.concatenate([[0,], new_labels])

        mapping_table[0]['move'] = 1
        label_start = mapping_table['start'][mapping_table['move'] > 0]
        label_start = label_start[label_start < ub].data

        idx = np.zeros(ub, dtype=np.int)
        idx[label_start] = np.arange(label_start.shape[0], dtype=np.int) + 1
        idx = fill_zeros_with_prev(idx)
        idx = idx.reshape((ml, chunk_len))[:,::downsample_factor]
        idx = np.apply_along_axis(remove_same, 1, idx)

        sig_labels = new_labels[idx]

    # Bad state isn't supported yet with raw models
    sig_bad = np.zeros((ml, chunk_len), dtype=bool)

    return new_inMat, sig_labels, sig_bad


def raw_chunk_worker(fn, chunk_len, kmer_len, min_length, trim, normalisation,
                downsample_factor, interpolation=False):
    """  Worker for creating labelled features from raw data

    :param fn: A filename to read from.
    :param chunk_len: Length on each chunk
    :param kmer_len: Kmer length for training
    :param min_length: Minumum number of samples before read can be considered.
    :param trim: Tuple (beginning, end) of number of samples to trim from read.
    :param normalisation: Normalisation method [per-chunk | per-read | none]
    :param downsample_factor: factor by which to downsample labels
    :param interpolation: interpolate sequence positions between those in
        mapping table
    """
    kmer_to_state = bio.kmer_mapping(kmer_len)

    try:
        with fast5.Reader(fn) as f5:
            mapping_table, att = f5.get_any_mapping_data('template')
            sig = f5.get_read(raw=True)
            sample_rate = f5.sample_rate
            start_sample = f5.get_read(raw=True, group=True).attrs['start_time']
    except Exception as e:
        sys.stderr.write('Failed to get mapping data from {}.\n{}\n'.format(fn, repr(e)))
        return None

    mapping_table = commensurate_mapping_to_raw(mapping_table, start_sample, sample_rate)

    map_start = mapping_table['start'][0]
    map_end = mapping_table['start'][-1] + mapping_table['length'][-1]
    sig_mapped = sig[map_start:map_end]
    mapping_table['start'] -= map_start

    sig_trim = util.trim_array(sig_mapped, *trim)

    if len(sig_trim) < min(chunk_len, min_length):
        sys.stderr.write('{} is too short.\n'.format(fn))
        return None

    new_inMat, sig_labels, sig_bad = raw_chunkify(sig_trim, mapping_table, chunk_len, kmer_len, normalisation, downsample_factor, interpolation)

    return (np.ascontiguousarray(new_inMat),
            np.ascontiguousarray(sig_labels),
            np.ascontiguousarray(sig_bad))


def raw_chunkify_with_identity_main(args):

    if not args.overwrite:
        if os.path.exists(args.output):
            print("Cowardly refusing to overwrite {}".format(args.output))
            sys.exit(1)

    fast5_files = fast5.iterate_fast5(args.input_folder, paths=True,
                                      limit=args.limit,
                                      strand_list=args.input_strand_list)

    print('* Processing data using', args.jobs, 'threads')

    kwarg_names = ['chunk_len', 'kmer_len', 'min_length', 'trim', 'normalisation', 'downsample_factor', 'interpolation']
    i = 0
    bad_list = []
    chunk_list = []
    label_list = []
    for res in imap_mp(raw_chunk_worker, fast5_files, threads=args.jobs,
                       unordered=True, fix_kwargs=util.get_kwargs(args, kwarg_names)):
        if res is not None:
            i = util.progress_report(i)

            (chunks, labels, bad_ev) = res

            chunk_list.append(chunks)
            label_list.append(labels)
            bad_list.append(bad_ev)

    if chunk_list == []:
        print("no chunks were produced", file=sys.stderr)
        sys.exit(1)
    else:
        print('\n* Writing out to HDF5')
        hdf5_attributes = {
            'chunk': args.chunk_len,
            'kmer': args.kmer_len,
            'trim': args.trim,
            'normalisation': args.normalisation,
            'downsample_factor': args.downsample_factor,
            'interpolation': args.interpolation
        }
        util.create_hdf5(args.output, args.blanks, hdf5_attributes, chunk_list, label_list, bad_list)


def raw_chunkify_with_remap_main(args):
    pass
