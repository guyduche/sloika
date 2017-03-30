#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

from Bio import SeqIO
try:
    from StringIO import StringIO
except:
    from io import StringIO
import argparse
import os
import sys

from untangled import fast5
from untangled.cmdargs import (AutoBool, FileExists, Maybe, Positive)
from untangled.iterators import imap_mp

from sloika import util


program_description = "Extract refereces from fast5 files"
parser = argparse.ArgumentParser(description=program_description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_strand_list', default=None, action=FileExists,
                    help='Strand summary file containing subset')
parser.add_argument('--jobs', default=1, metavar='n', type=Positive(int),
                    help='Number of threads to use when processing data')
parser.add_argument('--limit', default=None, type=Maybe(Positive(int)),
                    help='Limit number of reads to process')
parser.add_argument('--overwrite', default=False, action=AutoBool,
                    help='Whether to overwrite any output files')
parser.add_argument('--section', default='template',
                    choices=['template', 'complement'], help='Section to call')
parser.add_argument('input_folder', action=FileExists,
                    help='Directory containing single-read fast5 files')
parser.add_argument('output', help='Output fasta file')


def reference_extraction_worker(file_name, section):
    with fast5.Reader(file_name) as file_handle:
        try:
            fasta = file_handle.get_reference_fasta(section=section).decode('utf-8')
        except Exception as e:
            sys.stderr.write('No reference found for {}.\n{}\n'.format(file_name, repr(e)))
            return None

        iowrapper = StringIO(fasta)
        read_ref = str(next(SeqIO.parse(iowrapper, 'fasta')).seq)
        return (file_name, read_ref)


def main(argv):
    args = parser.parse_args(argv[1:])

    if not args.overwrite:
        if os.path.exists(args.output):
            print("Cowardly refusing to overwrite {}".format(args.output))
            sys.exit(1)

    fast5_files = fast5.iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                                      strand_list=args.input_strand_list)

    print('* Processing data using', args.jobs, 'threads')

    i = 0
    kwarg_names = ['section']
    mode = 'w' if sys.version_info.major == 3 else 'wb'
    with open(args.output, mode) as file_handle:
        for res in imap_mp(reference_extraction_worker, fast5_files, threads=args.jobs, unordered=True,
                           fix_kwargs=util.get_kwargs(args, kwarg_names)):
            if res is not None:
                i = util.progress_report(i)
                file_name, reference = res
                header = '>{}\n'.format(os.path.basename(os.path.splitext(file_name)[0]))
                file_handle.write(header)
                file_handle.write(reference + '\n')

if __name__ == '__main__':
    sys.exit(main(sys.argv[:]))
