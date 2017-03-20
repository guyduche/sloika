from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

import numpy as np
from sloika import util
from untangled import bio, fast5
from untangled.iterators import imap_mp
from untangled.maths import med_mad, studentise, mad


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
                downsample_factor, downsample_method="simple"):
    """  Worker for creating labelled features from raw data

    :param fn: A filename to read from.
    :param section: Section of read to process (template / complement)
    :param chunk_len: Length on each chunk
    :param kmer_len: Kmer length for training
    :param min_length: Minumum number of samples before read can be considered.
    :param trim: Tuple (beginning, end) of number of samples to trim from read.
    :param normalise: Do per-strand normalisation
    :param downsample_factor: factor by which to downsample labels
    :param downsample_method: method to use for downsampling, either
        "simple" or "interpolation"
    """
    kmer_to_state = bio.kmer_mapping(kmer_len)

    try:
        with fast5.Reader(fn) as f5:
            mapping_table, att = f5.get_any_mapping_data(section)
            sig = f5.get_read(raw=True)
            sample_rate = f5.sample_rate
            start_sample = f5.get_read(raw=True, group=True).attrs['start_time']
    except:
        sys.stderr.write('Failed to get mapping data from {}.\n{}\n'.format(fn, repr(e)))
        return None

    mapping_table['move'][0] = 1

    map_start_time = mapping_table['start'][0]
    map_start_sample = int(round(map_start_time * sample_rate - start_sample))
    map_end_time = mapping_table['start'][-1] + mapping_table['length'][-1]
    map_end_sample = int(round(map_end_time * sample_rate - start_sample))
    sig_mapped = sig[map_start_sample:map_end_sample]

    sig_trim = util.trim_array(sig_mapped, *trim)

    if len(sig_trim) < min(chunk_len, min_length):
        sys.stderr.write('{} is too short.\n'.format(fn))
        return None

    if normalise:
        loc, scale = med_mad(sig_trim)
        sig_trim = (sig_trim - loc) / scale

    #  Create feature array
    ml = len(sig_trim) // chunk_len
    sig_trim = sig_trim[:ml * chunk_len].reshape((ml, chunk_len, 1))

    if downsample_method == "simple":
        #  Create label array
        model_kmer_len = len(mapping_table['kmer'][0])
        ub = chunk_len * ml
        # Use rightmost middle kmer
        kl = (model_kmer_len - kmer_len + 1) // 2
        ku = kl + kmer_len
        new_labels = 1 + np.array(map(lambda k: kmer_to_state[k[kl : ku]],
                                      mapping_table['kmer'][mapping_table['move'] > 0]), dtype=np.int32)
        new_labels = np.concatenate([[0,], new_labels])

        label_start = (np.around(mapping_table['start'][mapping_table['move'] > 0] * sample_rate).astype(int)
                                      - start_sample - map_start)
        label_start = label_start[label_start < ub].data

        idx = np.zeros(ub, dtype=int)
        idx[label_start] = np.arange(label_start.shape[0]) + 1
        idx = fill_zeros_with_prev(idx)
        idx = idx.reshape((ml, chunk_len))[:,::downsample_factor]
        idx = np.apply_along_axis(remove_same, 1, idx)

        sig_labels = new_labels[idx]
    elif downsample_method == "interpolation":
        delta = 1.0 / sample_rate
        chunk_delta = delta * chunk_len
        t0 = (map_start + start_sample) / sample_rate
        tch = np.arange(t0, t0 + chunk_delta, downsample_factor * delta)
        t = np.array([tch + i * chunk_delta for i in range(ml)]).flatten()
        pos = interpolate_pos(mapping_table, att)(t, kmer_len)
        kmers = interpolate_labels(mapping_table, att)(t, kmer_len)
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


def raw_chunkify_with_identity_main(argv, parser):
    parser.add_argument('--downsample_factor', default=1, type=Positive(int),
                        help='Rate of label downsampling')
    parser.add_argument('--interpolation', default=False, action=AutoBool,
                        help='Interpolate reference sequence positions between mapped samples')
    args = parser.parse_args()
    pass


def raw_chunkify_with_remap_main(argv, parser):
    parser.add_argument('--compile', default=None, type=Maybe(str),
                        help='File output compiled model')
    parser.add_argument('--downsample_factor', default=1, type=Positive(int),
                        help='Rate of label downsampling')
    parser.add_argument('--interpolation', default=False, action=AutoBool,
                        help='Interpolate reference sequence positions between mapped samples')
    parser.add_argument('--min_prob', metavar='proportion', default=1e-5,
                        type=proportion, help='Minimum allowed probabiility for basecalls')
    parser.add_argument('--output_strand_list', default="strand_output_list.txt",
                        help='strand summary output file')
    parser.add_argument('--prior', nargs=2, metavar=('start', 'end'), default=(25.0, 25.0),
                        type=Maybe(NonNegative(float)), help='Mean of start and end positions')
    parser.add_argument('--slip', default=5.0, type=Maybe(NonNegative(float)),
                        help='Slip penalty')
    parser.add_argument('--stride', default=4, type=int,
                        help='Stride of the model used for remapping')

    parser.add_argument('model', action=FileExists, help='Pickled model file')
    parser.add_argument('references', action=FileExists,
                        help='Reference sequences in fasta format'

    args = parser.parse_args()
    pass
