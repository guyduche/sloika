import numpy as np
from sloika import features
from untangled import bio, fast5

_NBASE = 4

def kmers(files, section, batch_size, chunk_len, window, kmer_len, bad=False,
          trim=(0, 0), shuffle=True):
    """ Batch data together for kmer training

    :param files: A `set` of files to read
    :param section: Section of read to process (template / complement)
    :param batch_size: Size of batch of chunks
    :param chunk_len: Length on each chunk
    :param window: Length of window for features
    :param kmer_len: Kmer length for training
    :param shuffle: Shuffle order of files

    :yields: A tuple containing a 3D :class:`ndarray` of size
    (X, chunk_len, nfeatures) containing the features for the batch
    and a 2D :class`ndarray` of size (X, chunk_len) containing the
    associated labels.  1 <= X <= batch_size.
    """
    kmer_to_state = bio.kmer_mapping(1)
    trim_len = sum(trim)
    begin, end = trim
    end = None if end is 0 else -end

    pfiles = list(files)
    if shuffle:
        pfiles = np.random.permutation(pfiles)

    in_mat = labels = None
    for fn in pfiles:
        try:
            with fast5.Reader(fn) as f5:
                ev, _ = f5.get_any_mapping_data(section)
        except:
            continue
        if len(ev) <= trim_len + chunk_len + window:
            continue

        new_inMat = features.from_events(ev[begin : end])
        ml = len(new_inMat) // chunk_len
        new_inMat = new_inMat[:ml * chunk_len].reshape((ml, chunk_len, -1))

        model_kmer_len = len(ev['kmer'][0])
        l = begin
        u = l + chunk_len * ml
        kl = model_kmer_len // 2
        ku = kl + kmer_len
        new_labels = np.array(map(lambda k: kmer_to_state[k[kl : ku]], ev['kmer'][l:u]), dtype=np.int32)
        if bad:
            new_labels[np.logical_not(ev['good_emission'][l:u])] = _NBASE ** kmer_len
        new_labels = new_labels.reshape((ml, chunk_len))
        new_labels = new_labels[:, (window // 2) : -(window // 2)]

        in_mat = np.vstack((in_mat, new_inMat)) if in_mat is not None else new_inMat
        labels = np.vstack((labels, new_labels)) if labels is not None else new_labels
        while len(in_mat) > batch_size:
            yield (np.ascontiguousarray(in_mat[:batch_size].transpose((1,0,2))),
                   np.ascontiguousarray(labels[:batch_size].transpose()))
            in_mat = in_mat[batch_size:]
            labels = labels[batch_size:]


def transducer(files, section, batch_size, chunk_len, window,
               trim=(0, 0), shuffle=True):
    """ Batch dat together for transducer

    :param files: A `set` of files to read
    :param section: Section of read to process (template / complement)
    :param batch_size: Size of batch of chunks
    :param chunk_len: Length on each chunk
    :param window: Length of window for features
    :param trim: A tuple with number of events to trim off beginning and end
    :param shuffle: Shuffle order of files

    :yields: A tuple containing a 3D :class:`ndarray` of size
    (X, chunk_len, nfeatures) containing the features for the batch
    and a 2D :class`ndarray` of size (X, chunk_len) containing the
    associated labels.  1 <= X <= batch_size.
    """
    kmer_to_state = bio.kmer_mapping(1)
    trim_len = sum(trim)
    begin, end = trim
    end = None if end is 0 else -end

    pfiles = list(files)
    if shuffle:
        pfiles = np.random.permutation(pfiles)

    in_mat = labels = None
    for fn in pfiles:
        try:
            with fast5.Reader(fn) as f5:
                ev, _ = f5.get_any_mapping_data(section)
        except:
            continue
        if len(ev) <= trim_len + chunk_len + window:
            continue

        new_inMat = features.from_events(ev[begin : end])
        ml = len(new_inMat) // chunk_len
        new_inMat = new_inMat[:ml * chunk_len].reshape((ml, chunk_len, -1))

        kmer_len = len(ev['kmer'][0])
        l = begin
        u = l + chunk_len * ml
        kp = kmer_len // 2
        new_labels = np.array(map(lambda k: kmer_to_state[k[kp]], ev['kmer'][l:u]), dtype=np.int32)
        new_labels[np.ediff1d(ev['seq_pos'][l:u], to_begin=1) == 0] = _NBASE
        new_labels = new_labels.reshape((ml, chunk_len))
        new_labels = new_labels[:, (window // 2) : -(window // 2)]

        in_mat = np.vstack((in_mat, new_inMat)) if in_mat is not None else new_inMat
        labels = np.vstack((labels, new_labels)) if labels is not None else new_labels
        while len(in_mat) > batch_size:
            yield (np.ascontiguousarray(in_mat[:batch_size].transpose((1,0,2))),
                   np.ascontiguousarray(labels[:batch_size].transpose()))
            in_mat = in_mat[batch_size:]
            labels = labels[batch_size:]
