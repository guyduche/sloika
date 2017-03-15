from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

import numpy as np
import sys

from untangled import bio, fast5
from untangled.maths import mad

from sloika import util
from sloika.variables import nstate


def init_worker(model):
    import pickle
    global calc_post
    with open(model, 'rb') as fh:
        calc_post = pickle.load(fh)


def decode_post(post, args, eta=1e-10):
    from sloika import decode
    assert post.shape[2] == nstate(args.kmer_len, transducer=args.transducer, bad_state=args.bad)
    post = decode.prepare_post(post, min_prob=args.min_prob, drop_bad=args.bad and not args.transducer)
    if args.transducer:
        score, call = decode.viterbi(post, args.kmer_len, skip_pen=args.skip)
    else:
        trans = olddecode.estimate_transitions(post, trans=args.trans)
        score, call = olddecode.decode_profile(post, trans=np.log(eta + trans), log=False)
    return score, call


def events_worker(args, fn):
    from sloika import features
    try:
        with fast5.Reader(fn) as f5:
            ev = f5.get_section_events(args.section, analysis=args.segmentation)
            sn = f5.filename_short
    except Exception as e:
        sys.stderr.write("Error getting events for section {!r} in file {}\n{!r}\n".format(args.section, fn, e))
        return None

    ev = util.trim_array(ev, args.trim)
    if ev.size == 0:
        sys.stderr.write("Read too short in file {}\n".format(fn))
        return None

    inMat = features.from_events(ev, tag='')[:, None, :]
    score, call = decode_post(calc_post(inMat), args)

    return sn, score, call, inMat.shape[0]


def raw_worker(args, fn):
    from sloika import batch, config
    try:
        with fast5.Reader(fn) as f5:
            signal = f5.get_read(raw=True)
            sn = f5.filename_short
    except Exception as e:
        sys.stderr.write("Error getting raw data for file {}\n{!r}\n".format(fn, e))
        return None

    signal = batch.trim_open_pore(signal, args.open_pore_fraction)
    signal = util.trim_array(signal, args.trim)
    if signal.size == 0:
        sys.stderr.write("Read too short in file {}\n".format(fn))
        return None

    inMat = (signal - np.median(signal)) / mad(signal)
    inMat = inMat[:, None, None].astype(config.sloika_dtype)
    score, call = decode_post(calc_post(inMat), args)

    return sn, score, call, inMat.shape[0]


class SeqPrinter(object):

    def __init__(self, kmerlen, datatype="events", transducer=False, fname=None):
        self.kmers = bio.all_kmers(kmerlen)
        self.transducer = transducer
        self.datatype = datatype

        if fname is None:
            self.fh = sys.stdout
            self.close_fh = False
        else:
            self.fh = open(fname, 'w')
            self.close_fh = True

    def __del__(self):
        if self.close_fh:
            self.fh.close()

    def write(self, read_name, score, call, nev):
        kmer_path = [self.kmers[i] for i in call]
        seq = bio.kmers_to_sequence(kmer_path, always_move=self.transducer)
        self.fh.write(">{} score {:.0f}, {} {} to {} bases\n".format(read_name, score,
                                                                     nev, self.datatype, len(seq)))
        self.fh.write(seq + '\n')
        return len(seq)
