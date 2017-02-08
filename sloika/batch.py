from __future__ import print_function
import h5py
import numpy as np
import numpy.lib.recfunctions as nprf
import sys

from Bio import SeqIO

import sloika

from untangled import bio, fast5
from untangled.maths import med_mad

def chunkify(ev, chunk_len, kmer_len, use_scaled, normalise):
    ml = len(ev) // chunk_len
    ub = ml * chunk_len

    #
    # we may pass bigger range to the function below than we would
    # actually use later, so that features could be studentized using
    # moments computed using this bigger range
    #
    new_inMat = sloika.features.from_events(ev, tag='' if use_scaled else 'scaled_',
                                     normalise=normalise)
    ev = ev[0 : ub]
    new_inMat = new_inMat[0 : ub].reshape((ml, chunk_len, -1))

    #
    # 'model' in the name 'model_kmer_len' refers to the model that was used
    # to map the reads read from the fast5 file above
    #
    model_kmer_len = len(ev['kmer'][0])
    # Use rightmost middle kmer
    kl = (model_kmer_len - kmer_len + 1) // 2
    ku = kl + kmer_len
    kmer_to_state = bio.kmer_mapping(kmer_len)
    new_labels = 1 + np.array(map(lambda k: kmer_to_state[k[kl : ku]],
                                  ev['kmer']), dtype=np.int32)

    new_labels = new_labels.reshape(ml, chunk_len)
    change = ev['seq_pos'].reshape(ml, chunk_len)
    change = np.apply_along_axis(np.ediff1d, 1, change, to_begin=1)
    new_labels[change == 0] = 0

    new_bad = np.logical_not(ev['good_emission'])
    new_bad = new_bad.reshape(ml, chunk_len)

    assert sloika.util.is_contiguous(new_inMat)
    assert sloika.util.is_contiguous(new_labels)
    assert sloika.util.is_contiguous(new_bad)

    return new_inMat, new_labels, new_bad


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

    return chunkify(ev, chunk_len, kmer_len, use_scaled, normalise)

def init_chunk_remap_worker(model, fasta, kmer_len):
    import cPickle
    global calc_post, kmer_to_state, references
    with open(model, 'r') as fh:
        calc_post = cPickle.load(fh)

    references = dict()
    with open(fasta, 'r') as fh:
        for ref in SeqIO.parse(fh, 'fasta'):
            refseq = str(ref.seq)
            if 'N' not in refseq:
                references[ref.id] = refseq
    sys.stderr.write('Read {} references from {}.\n'.format(len(references), fasta))

    kmer_to_state = bio.kmer_mapping(kmer_len)


def remap(read_ref, ev, min_prob, transducer, kmer_len, prior, slip):
    inMat = sloika.features.from_events(ev, tag='')
    inMat = np.expand_dims(inMat, axis=1)
    post = sloika.decode.prepare_post(calc_post(inMat), min_prob=min_prob, drop_bad=(not transducer))

    kmers = np.array(bio.seq_to_kmers(read_ref, kmer_len))
    seq = map(lambda k: kmer_to_state[k] + 1, kmers)
    prior0 = None if prior[0] is None else sloika.util.geometric_prior(len(seq), prior[0])
    prior1 = None if prior[1] is None else sloika.util.geometric_prior(len(seq), prior[1], rev=True)

    score, path = sloika.transducer.map_to_sequence(post, seq, slip=slip,
                                             prior_initial=prior0,
                                             prior_final=prior1, log=False)

    ev = nprf.append_fields(ev, ['seq_pos', 'kmer', 'good_emission'],
                            [path, kmers[path], np.repeat(True, len(ev))])

    return (score, ev, path, seq)

def chunk_remap_worker(fn, trim, min_prob, transducer, kmer_len, prior, slip, chunk_len, use_scaled, normalise):
    try:
        with fast5.Reader(fn) as f5:
            ev = f5.get_read()
            sn = f5.filename_short
    except:
        sys.stderr.write('Failure reading events from {}.\n'.format(fn))
        return None

    try:
        read_ref = references[sn]
    except:
        sys.stderr.write('No reference found for {}.\n'.format(fn))
        return None

    if len(ev) <= sum(trim):
        sys.stderr.write('{} with {} events is too short.\n'.format(fn, len(ev)))
        return None

    begin, end = trim
    end = None if end is 0 else -end
    ev = ev[begin : end]

    (score, ev, path, seq) = remap(read_ref, ev, min_prob, transducer, kmer_len, prior, slip)
    ev = ev[50: -10]  -- TODO(semen): remove me -- only needed to mediate the transition to new behavior
    (chunks, labels, bad_ev) = chunkify(ev, chunk_len, kmer_len, use_scaled, normalise)

    return sn + '.fast5', score, len(ev), path, seq, chunks, labels, bad_ev

