from __future__ import print_function
import numpy as np
from sloika import features, util
from untangled import bio, fast5
from untangled.maths import med_mad


def chunk_worker(fn, section, chunk_len, kmer_len, min_length, trim, use_scaled,
                       normalise):
    """ Chunkifies data for training

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
    indicating bad events.  1 <= X <= batch_size.
    """

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
    ub = chunk_len * ml
    new_inMat = new_inMat[:ub].reshape((ml, chunk_len, -1))

    model_kmer_len = len(ev['kmer'][0])
    # Use rightmost middle kmer
    kl = (model_kmer_len - kmer_len + 1) // 2
    ku = kl + kmer_len
    kmer_to_state = bio.kmer_mapping(kmer_len)
    new_labels = 1 + np.array(map(lambda k: kmer_to_state[k[kl : ku]],
                                  ev['kmer'][:ub]), dtype=np.int32)

    new_labels = new_labels.reshape((ml, chunk_len))
    change = ev['seq_pos'][:ub].reshape((ml, chunk_len))
    change = np.apply_along_axis(np.ediff1d, 1, change, to_begin=1)
    new_labels[change == 0] = 0

    new_bad = np.logical_not(ev['good_emission'][:ub])
    new_bad = new_bad.reshape(ml, chunk_len)

    assert util.is_contiguous(new_inMat)
    assert util.is_contiguous(new_labels)
    assert util.is_contiguous(new_bad)

    return new_inMat, new_labels, new_bad
