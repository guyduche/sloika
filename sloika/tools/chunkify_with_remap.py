import argparse
import pickle
import os
import posixpath
import sys
import time
import numpy as np

from untangled import fast5
from untangled.iterators import imap_mp

from sloika import helpers, batch, util


def create_output_strand_file(output_strand_list_entries, output_file_name):
    output_strand_list_entries.sort()

    with open(output_file_name, "w") as sl:
        sl.write(u'\t'.join(['filename', 'nev', 'score', 'nstay', 'seqlen', 'start', 'end']) + u'\n')
        for strand_data in output_strand_list_entries:
            sl.write('\t'.join([str(x) for x in strand_data]) + '\n')


def chunkify_with_remap_main(args):

    if not args.overwrite:
        if os.path.exists(args.output):
            print("Cowardly refusing to overwrite {}".format(args.output))
            sys.exit(1)
        if os.path.exists(args.output_strand_list):
            print("Cowardly refusing to overwrite {}".format(args.output_strand_list))
            sys.exit(2)

    fast5_files = fast5.iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                                      strand_list=args.input_strand_list)

    references = util.fasta_file_to_dict(args.references)

    print('* Processing data using', args.jobs, 'threads')

    kwarg_names = ['trim', 'min_prob', 'kmer_len', 'min_length',
                   'prior', 'slip', 'chunk_len', 'use_scaled', 'normalisation',
                   'section', 'segmentation', 'alphabet']
    kwargs = util.get_kwargs(args, kwarg_names)
    kwargs['references'] = references

    i = 0
    compiled_file = helpers.compile_model(args.model, args.compile)
    output_strand_list_entries = []
    bad_list = []
    chunk_list = []
    label_list = []
    for res in imap_mp(batch.chunk_remap_worker, fast5_files, threads=args.jobs,
                       fix_kwargs=kwargs, unordered=True, init=batch.init_chunk_remap_worker,
                       initargs=[compiled_file]):
        if res is not None:
            i = util.progress_report(i)

            read, score, nev, path, seq, chunks, labels, bad_ev = res

            chunk_list.append(chunks)
            label_list.append(labels)
            bad_list.append(bad_ev)
            output_strand_list_entries.append([read, nev, -score / nev,
                                               np.sum(np.ediff1d(path, to_begin=1) == 0),
                                               len(seq), min(path), max(path)])

    if compiled_file != args.compile:
        os.remove(compiled_file)

    if chunk_list == []:
        print("no chunks were produced", file=sys.stderr)
        sys.exit(1)
    else:
        print('\n* Creating HDF5 file')
        hdf5_attributes = {
            'chunk': args.chunk_len,
            'input_type': 'events',
            'kmer': args.kmer_len,
            'normalisation': args.normalisation,
            'scaled': args.use_scaled,
            'section': args.section,
            'trim': args.trim,
            'alphabet': args.alphabet,
        }
        util.create_labelled_chunks_hdf5(args.output, args.blanks, hdf5_attributes, chunk_list, label_list, bad_list)

        print('\n* Creating output strand file')
        create_output_strand_file(output_strand_list_entries, args.output_strand_list)
