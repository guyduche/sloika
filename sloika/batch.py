from __future__ import print_function
import numpy as np
from sloika import features
from untangled import bio, fast5
from untangled.maths import med_mad


#
# TODO(semen): this function appears to be unused, but my notes suggest it should be
#
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
    nchunk = len(position) // chunk
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


def batch_chunk_worker(fn, section, chunk_len, kmer_len, min_length, trim, use_scaled,
                       normalise):
    """ Batcher and chunkifier of data for training

    :param fn: A filename to read from.
    :param section: Section of read to process (template / complement)
    :param chunk_len: Length of each chunk
    :param kmer_len: Kmer length for training
    :param min_length: Minimum number of events before read can be considered.
    :param trim: Tuple (beginning, end) of number of events to trim from read.
    :param use_scaled: Use prescaled event statistics
    :param normalise: Do per-strand normalisation

    :yields: A tuple containing a 3D :class:`ndarray` of size
    (X, chunk_len, nfeatures) containing the features for the batch,
    a 2D :class:`ndarray` of size (X, chunk_len) containing the
    associated labels, and a 2D :class:`ndarray` of size (X, chunk_len)
    containing bad events.  1 <= X <= batch_size.
    """

    kmer_to_state = bio.kmer_mapping(kmer_len)

    try:
        with fast5.Reader(fn) as f5:
            ev, _ = f5.get_any_mapping_data(section)
    except:
        return None, None, None

    if len(ev) < sum(trim) + chunk_len or len(ev) < min_length:
        return None, None, None
    begin, end = trim
    end = None if end is 0 else -end
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

    new_labels = new_labels.reshape((ml, chunk_len))
    change = ev['seq_pos'][:ub].reshape((ml, chunk_len))
    change = np.apply_along_axis(np.ediff1d, 1, change, to_begin=1)
    new_labels[change == 0] = 0

    new_bad = np.logical_not(ev['good_emission'][:ub])
    new_bad = new_bad.reshape(ml, chunk_len)

    return new_inMat, new_labels, new_bad
