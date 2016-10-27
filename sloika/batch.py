from __future__ import print_function
import numpy as np
from sloika import features
from untangled import bio, fast5
from untangled.iterators import imap_mp
from untangled.maths import med_mad

_NBASE = 4

def filter_by_rate(position, chunk, time=None, fact=3.0):
    """  Filter chunks using sequencing rate from mapping

    Fits a linear regression through the mapping of events and indicates
    whether any regions have a unusual bps

    :param position: A :class:`ndarray` containing positions mapped to.
    :param chunk:
    :param time: A :class:`ndarray` with time of each event or None.  If None,
    the index of the event is used.
    :param fact: Number of standard deviations after which a slope will be
    considered bad.

    :returns: A :class:`ndarray` contain a boolean of whether chunk is good
    """
    assert time is None or len(position) == len(time)
    nchunk = len(position)  // chunk
    chunk_idx = chunk * np.arange(nchunk)
    delta_pos = position[chunk_idx + chunk - 1] - position[chunk_idx]
    if time is None:
        delta_time = chunk
    else:
        delta_time = time[chunk_idx + chunk - 1] - time[chunk_idx]

    bps = delta_pos / delta_time
    #  Determine accept / reject regions
    centre, thresh = med_mad(bps)
    bps -= centre
    thresh *= fact
    return np.logical_and(bps < thresh, bps > -thresh)


def _kmer_worker(fn, section, chunk_len, kmer_len, min_length, trim, use_scaled,
                 normalise):
    """  Worker for reading kmer-overlap data

    :param fn: A filename to read from.
    :param section: Section of read to process (template / complement)
    :param chunk_len: Length on each chunk
    :param kmer_len: Kmer length for training
    :param min_length: Minumum number of events before read can be considered.
    :param trim: Tuple (beginning, end) of number of events to trim from read.
    :param use_scaled: Use prescaled event statistics
    :param normalise: Do per-strand normalisation
    """
    kmer_to_state = bio.kmer_mapping(kmer_len)
    begin, end = trim
    end = None if end is 0 else -end

    try:
        with fast5.Reader(fn) as f5:
            ev, _ = f5.get_any_mapping_data(section)
    except:
        return fn, None, None, None

    if len(ev) < sum(trim) + chunk_len or len(ev) < min_length:
        return fn, None, None, None
    ev = ev[begin : end]

    new_inMat = features.from_events(ev, tag='' if use_scaled else 'scaled_',
                                     normalise=normalise)
    ml = len(new_inMat) // chunk_len
    new_inMat = new_inMat[:ml * chunk_len].reshape((ml, chunk_len, -1))

    model_kmer_len = len(ev['kmer'][0])
    ub = chunk_len * ml
    # Use rightmost middle kmer
    kl = (model_kmer_len - kmer_len + 1) // 2
    ku = kl + kmer_len
    new_labels = 1 + np.array(map(lambda k: kmer_to_state[k[kl : ku]],
                                  ev['kmer'][:ub]), dtype=np.int32)

    new_labels[np.ediff1d(ev['seq_pos'][:ub], to_begin=1) == 0] = 0
    new_labels = new_labels.reshape((ml, chunk_len))

    new_bad  = np.logical_not(ev['good_emission'][:ub])
    new_bad = new_bad.reshape(ml, chunk_len)

    return fn, new_inMat, new_labels, new_bad


def kmers(files, section, chunk_len, kmer_len, min_length=0, trim=(0, 0),
          use_scaled=False, normalise=True):
    """ Batch data together for kmer training

    :param files: A `set` of files to read
    :param section: Section of read to process (template / complement)
    :param chunk_len: Length on each chunk
    :param kmer_len: Kmer length for training
    :param min_length: Minumum number of events before read can be considered.
    :param trim: Tuple (beginning, end) of number of events to trim from read.
    :param use_scaled: Use prescaled event statistics
    :param normalise: Do per-strand normalisation

    :yields: A tuple containing a 3D :class:`ndarray` of size
    (X, chunk_len, nfeatures) containing the features for the batch
    and a 2D :class`ndarray` of size (X, chunk_len) containing the
    associated labels.  1 <= X <= batch_size.
    """

    pfiles = list(files)

    wargs = {'chunk_len' : chunk_len,
             'kmer_len' : kmer_len,
             'min_length' : min_length,
             'normalise' : normalise,
             'section' : section,
             'trim' : trim,
             'use_scaled' : use_scaled
            }

    for fn, chunks, labels, bad_ev in imap_mp(_kmer_worker, pfiles, threads=8,
                                              fix_kwargs=wargs):
        if chunks is None or labels is None:
            continue

        yield (np.ascontiguousarray(chunks),
               np.ascontiguousarray(labels),
               np.ascontiguousarray(bad_ev))
