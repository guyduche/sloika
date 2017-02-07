from __future__ import print_function

import argparse
import cPickle
import h5py
import numpy as np
import numpy.lib.recfunctions as nprf
import posixpath
import sys
import time
import os

from Bio import SeqIO

from untangled import bio
from untangled.cmdargs import (AutoBool, FileExists, Maybe, NonNegative,
                               proportion, Positive, Vector, FileAbsent)
from untangled import fast5
from untangled.iterators import imap_mp

from sloika import helpers
from sloika.util import geometric_prior, get_kwargs


def init_worker(model, fasta, kmer_len):
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


def chunk_remap_worker(fn, trim, min_prob, transducer, kmer_len, prior, slip):
    from sloika import decode, features, transducer

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

    inMat = features.from_events(ev, tag='')
    inMat = np.expand_dims(inMat, axis=1)
    post = decode.prepare_post(calc_post(inMat), min_prob=min_prob, drop_bad=(not transducer))

    kmers = np.array(bio.seq_to_kmers(read_ref, kmer_len))
    seq = map(lambda k: kmer_to_state[k] + 1, kmers)
    prior0 = None if prior[0] is None else geometric_prior(len(seq), prior[0])
    prior1 = None if prior[1] is None else geometric_prior(len(seq), prior[1], rev=True)

    score, path = transducer.map_to_sequence(post, seq, slip=slip,
                                             prior_initial=prior0,
                                             prior_final=prior1, log=False)

    with h5py.File(fn, 'r+') as h5:
        #  A lot of messy and somewhat unnecessary work to make compatible with fast5 reader
        ds = '/Analyses/AlignToRef_000/CurrentSpaceMapped_template/Events'
        gs = '/Analyses/AlignToRef_000/Summary/current_space_map_template'
        gs2 = '/Analyses/Alignment_000/Summary/genome_mapping_template'
        fs = '/Analyses/Alignment_000/Aligned_template/Fasta'
        ev = nprf.append_fields(ev, ['seq_pos', 'kmer', 'good_emission'],
                                [path, kmers[path], np.repeat(True, len(ev))])

        if ds in h5:
            del h5[ds]
        h5.create_dataset(ds, data=ev)
        h5[ds].attrs['direction'] = '+'
        h5[ds].attrs['ref_start'] = 0
        h5[ds].attrs['ref_stop'] = len(read_ref)

        if gs in h5:
            del h5[gs]
        h5.create_group(gs)
        h5[gs].attrs['direction'] = '+'
        h5[gs].attrs['genome_start'] = 0
        h5[gs].attrs['genome_end'] = len(read_ref)
        h5[gs].attrs['genome'] = 'pseudo'
        h5[gs].attrs['num_skips'] = 0
        h5[gs].attrs['num_stays'] = 0

        if gs2 in h5:
            del h5[gs2]
        h5.create_group(gs2)
        h5[gs2].attrs['genome'] = 'pseudo'

        refdat = '>pseudo\n' + read_ref
        if fs in h5:
            del h5[fs]
        h5.create_dataset(fs, data=refdat)

    return sn + '.fast5', score, len(ev), path, seq


def chunkify_with_remap_main(argv, parser):
    parser.add_argument('--compile', default=None, type=Maybe(str),
                        help='File output compiled model')
    parser.add_argument('--min-prob', metavar='proportion', default=1e-5,
                        type=proportion, help='Minimum allowed probabiility for basecalls')
    parser.add_argument('--prior', nargs=2, metavar=('start', 'end'), default=(25.0, 25.0),
                        type=Maybe(NonNegative(float)), help='Mean of start and end positions')
    parser.add_argument('--slip', default=5.0, type=Maybe(NonNegative(float)),
                        help='Slip penalty')
    parser.add_argument('--strand-input-list', default=None, action=FileExists,
                        help='strand summary file containing subset')
    parser.add_argument('--transducer', default=True, action=AutoBool,
                        help='Model is transducer')
    parser.add_argument('--strand-output-list', default="strand_output_list.txt", action=FileAbsent,
                        help='strand summary output file')
    parser.add_argument('model', action=FileExists, help='Pickled model file')
    parser.add_argument('references', action=FileExists,
                        help='Reference sequences in fasta format')
    parser.add_argument('input_folder', action=FileExists,
                        help='Directory containing single-read fast5 files')

    args = parser.parse_args(argv)

    compiled_file = helpers.compile_model(args.model, args.compile)

    fast5_files = fast5.iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                                      strand_list=args.strand_input_list)

    print('* Processing data using', args.threads, 'threads')

    kwarg_names = ['trim', 'min_prob', 'transducer', 'kmer_len', 'prior', 'slip']
    strands_list = []
    for res in imap_mp(chunk_remap_worker, fast5_files, threads=args.threads, fix_kwargs=get_kwargs(args,kwarg_names),
                       unordered=True, init=init_worker, initargs=[compiled_file, args.references, args.kmer_len]):
        if res is not None:
            read, score, nev, path, seq = res
            strands_list.append([read, nev, -score / nev, np.sum(np.ediff1d(path, to_begin=1) == 0),
                                 len(seq), min(path), max(path)])

    strands_list.sort()
    with open(args.strand_output_list, "w") as sl:
        sl.write('\t'.join(['filename', 'nev', 'score', 'nstay', 'seqlen', 'start', 'end']) + '\n')
        for strand_data in strands_list:
            sl.write('\t'.join(map(lambda x: str(x), strand_data)) + '\n')

    if compiled_file != args.compile:
        os.remove(compiled_file)
