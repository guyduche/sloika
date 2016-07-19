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

def _kmer_worker(fn, section, chunk_len, window, kmer_len, trim, use_scaled, bad):
    """  Worker for reading kmer-overlap data

    :param fn: Fast5 filename
    :param section: Section of read to process (template / complement)
    :param chunk_len: Length on each chunk
    :param window: Length of window for features
    :param kmer_len: Kmer length for training
    :param trim: Tuple of number of events to trim from start and end
    :param use_scaled: Use prescaled event statistics
    :param bad: Have bad events a separate state
    """
    kmer_to_state = bio.kmer_mapping(1)
    begin, end = trim
    end = None if end is 0 else -end

    try:
        with fast5.Reader(fn) as f5:
            ev, _ = f5.get_any_mapping_data(section)
    except:
        return fn, None, None
    ev = ev[trim[0] : -trim[1]]

    new_inMat = features.from_events(ev, tag='' if use_scaled else 'scaled_')
    ml = len(new_inMat) // chunk_len
    new_inMat = new_inMat[:ml * chunk_len].reshape((ml, chunk_len, -1))

    model_kmer_len = len(ev['kmer'][0])
    ub = chunk_len * ml
    kl = model_kmer_len // 2
    ku = kl + kmer_len
    new_labels = np.array(map(lambda k: kmer_to_state[k[kl : ku]], ev['kmer'][:ub]),
                          dtype=np.int32)
    if bad:
        new_labels[np.logical_not(ev['good_emission'][:ub])] = _NBASE ** kmer_len
    new_labels = new_labels.reshape((ml, chunk_len))
    new_labels = new_labels[:, (window // 2) : -(window // 2)]

    return fn, new_inMat, new_labels

def kmers(files, section, batch_size, chunk_len, window, kmer_len, bad=False,
          trim=(0, 0), shuffle=True, use_scaled=False):
    """ Batch data together for kmer training

    :param files: A `set` of files to read
    :param section: Section of read to process (template / complement)
    :param batch_size: Size of batch of chunks
    :param chunk_len: Length on each chunk
    :param window: Length of window for features
    :param kmer_len: Kmer length for training
    :param shuffle: Shuffle order of files
    :param use_scaled: Use prescaled event statistics

    :yields: A tuple containing a 3D :class:`ndarray` of size
    (X, chunk_len, nfeatures) containing the features for the batch
    and a 2D :class`ndarray` of size (X, chunk_len) containing the
    associated labels.  1 <= X <= batch_size.
    """

    pfiles = list(files)
    if shuffle:
        pfiles = np.random.permutation(pfiles)

    in_mat = labels = None
    wargs = {'bad' : bad,
             'chunk_len' : chunk_len,
             'kmer_len' : kmer_len,
             'section' : section,
             'trim' : trim,
             'use_scaled' : use_scaled,
             'window' : window
            }

    for fn, new_inMat, new_labels in imap_mp(_kmer_worker, pfiles, threads=2, fix_kwargs=wargs):
        if new_inMat is None:
            assert new_labels is None
            continue
        assert new_labels is not None

        in_mat = np.vstack((in_mat, new_inMat)) if in_mat is not None else new_inMat
        labels = np.vstack((labels, new_labels)) if labels is not None else new_labels
        if len(in_mat) > batch_size:
            idx = np.random.permutation(len(in_mat))
            in_mat = in_mat[idx]
            labels = labels[idx]
            while len(in_mat) > batch_size:
                yield (np.ascontiguousarray(in_mat[:batch_size].transpose((1,0,2))),
                    np.ascontiguousarray(labels[:batch_size].transpose()))
                in_mat = in_mat[batch_size:]
                labels = labels[batch_size:]


def _transducer_worker(fn, section, chunk_len, window, filter_chunks, use_scaled, kmer_len=1):
    """ Worker to batch data together for transducer

    :param files: A `set` of files to read
    :param section: Section of read to process (template / complement)
    :param chunk_len: Length on each chunk
    :param window: Length of window for features
    :param filter_chunks: Filter by mapping slope?
    :param use_scaled: Use prescaled event statistics
    :param kmer_len: length of kmer

    :yields: A tuple containing a 3D :class:`ndarray` of size
    (X, chunk_len, nfeatures) containing the features for the batch
    and a 2D :class`ndarray` of size (X, chunk_len) containing the
    associated labels.  1 <= X <= batch_size.
    """
    kmer_to_state = bio.kmer_mapping(kmer_len)

    try:
        with fast5.Reader(fn) as f5:
            ev, _ = f5.get_any_mapping_data(section)
    except:
        return fn, None, None

    new_inMat = features.from_events(ev, tag='' if use_scaled else 'scaled_')
    ml = len(new_inMat) // chunk_len
    new_inMat = new_inMat[:ml * chunk_len].reshape((ml, chunk_len, -1))

    mapped_kmer_len = len(ev['kmer'][0])
    ub = chunk_len * ml
    k0 = (mapped_kmer_len - kmer_len + 1) // 2
    new_labels = np.array(map(lambda k: kmer_to_state[k[k0 : k0 + kmer_len]], ev['kmer'][:ub]), dtype=np.int32)
    new_labels[np.ediff1d(ev['seq_pos'][:ub], to_begin=1) == 0] = _NBASE
    new_labels = new_labels.reshape((ml, chunk_len))
    new_labels = new_labels[:, (window // 2) : -(window // 2)]

    if filter_chunks:
        accept = filter_by_rate(ev['seq_pos'], chunk_len, time=ev['start'])
        new_inMat = new_inMat[accept]
        new_labels = new_labels[accept]

    return fn, new_inMat, new_labels


def transducer(files, section, batch_size, chunk_len, window, filter_chunks=True,
               shuffle=True, use_scaled=False, kmer_len=1):
    """ Batch data together for transducer

    :param files: A `set` of files to read
    :param section: Section of read to process (template / complement)
    :param batch_size: Size of batch of chunks
    :param chunk_len: Length on each chunk
    :param window: Length of window for features
    :param filter_chunks: Filter by mapping slope?
    :param shuffle: Shuffle order of files
    :param use_scaled: Use prescaled event statistics
    :param kmer_len: length of kmer

    :yields: A tuple containing a 3D :class:`ndarray` of size
    (X, chunk_len, nfeatures) containing the features for the batch
    and a 2D :class`ndarray` of size (X, chunk_len) containing the
    associated labels.  1 <= X <= batch_size.
    """
    pfiles = list(files)
    if shuffle:
        pfiles = np.random.permutation(pfiles)

    in_mat = labels = None
    wargs = {'chunk_len' : chunk_len,
             'filter_chunks' : filter_chunks,
             'kmer_len' : kmer_len,
             'section' : section,
             'use_scaled' : use_scaled,
             'window' : window
            }
    for fn, new_inMat, new_labels in imap_mp(_transducer_worker, pfiles, threads=2, fix_kwargs=wargs):
        if new_inMat is None:
            assert new_labels is None
            continue
        assert new_labels is not None

        in_mat = np.vstack((in_mat, new_inMat)) if in_mat is not None else new_inMat
        labels = np.vstack((labels, new_labels)) if labels is not None else new_labels
        if len(in_mat) > batch_size:
            idx = np.random.permutation(len(in_mat))
            in_mat = in_mat[idx]
            labels = labels[idx]
            while len(in_mat) > batch_size:
                yield (np.ascontiguousarray(in_mat[:batch_size].transpose((1,0,2))),
                    np.ascontiguousarray(labels[:batch_size].transpose()))
                in_mat = in_mat[batch_size:]
                labels = labels[batch_size:]
