#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

import argparse
import sys

from untangled.cmdargs import (AutoBool, FileAbsent, FileExists, Maybe,
                               NonNegative, Positive, proportion)

from sloika.tools.chunkify_with_identity import chunkify_with_identity_main
from sloika.tools.chunkify_with_remap import chunkify_with_remap_main
from sloika import batch


def common_parser(argv, commands):
    program_name = argv[0]
    command_name = argv[1]
    program_description = commands.get_description(command_name)

    parser = argparse.ArgumentParser(prog=program_name + " " + command_name,
                                     description=program_description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--blanks', metavar='proportion', default=0.7,
                        type=proportion, help='Maximum proportion of blanks in labels')
    parser.add_argument('--chunk_len', default=500, metavar='events', type=Positive(int),
                        help='Length of each read chunk')
    parser.add_argument('--input_strand_list', default=None, action=FileExists,
                        help='strand summary file containing subset')
    parser.add_argument('--jobs', default=1, metavar='n', type=Positive(int),
                        help='Number of threads to use when processing data')
    parser.add_argument('--kmer_len', default=5, metavar='k', type=Positive(int),
                        help='Length of kmer to estimate')
    parser.add_argument('--limit', default=None, type=Maybe(Positive(int)),
                        help='Limit number of reads to process')
    parser.add_argument('--min_length', default=1200, metavar='events',
                        type=Positive(int), help='Minimum events in acceptable read')
    parser.add_argument('--normalisation', default=batch.DEFAULT_NORMALISATION, choices=batch.AVAILABLE_NORMALISATIONS,
                        help='Whether to perform studentisation and with what scope')
    parser.add_argument('--overwrite', default=False, action=AutoBool,
                        help='Whether to overwrite any output files')
    parser.add_argument('--section', default='template',
                        choices=['template', 'complement'], help='Section to call')
    parser.add_argument('--trim', default=(50, 10), nargs=2, type=NonNegative(int),
                        metavar=('beginning', 'end'),
                        help='Number of events to trim off start and end')
    parser.add_argument('--use_scaled', default=False, action=AutoBool,
                        help='Train from scaled event statistics')

    parser.add_argument('input_folder', action=FileExists,
                        help='Directory containing single-read fast5 files')
    parser.add_argument('output', help='Output HDF5 file')

    return (argv[2:], parser)


class Commands(object):

    def __init__(self, commands):
        self.commands = commands

    def __repr__(self):
        names = sorted(self.commands.keys())
        descriptions = [self.get_description(name) for name in names]
        name_description_pairs = list(zip(names, descriptions))

        def show_pair(t):
            return "%10s -- %s" % t
        return 'Available commands:\n\t' + '\n\t'.join(map(show_pair, name_description_pairs))

    def get_action(self, command_name):
        return self.commands[command_name][0]

    def get_description(self, command_name):
        return self.commands[command_name][1]


def main(argv):

    commands = Commands({
        'identity': (chunkify_with_identity_main, "Create HDF file from reads as is"),
        'remap': (chunkify_with_remap_main, "Create HDF file remapping reads on the fly using transducer network")
    })

    if 1 == len(argv):
        print(commands)
    else:
        command_name = argv[1]
        try:
            command_action = commands.get_action(command_name)
        except:
            print('Unsupported command {!r}'.format(command_name))
            print(commands)
            sys.exit(1)

        try:
            return command_action(*common_parser(argv, commands))
        except SystemExit as e:
            sys.exit(e.code)
        except:
            print('Exception when running command {!r}'.format(command_name))
            raise


if __name__ == '__main__':
    sys.exit(main(sys.argv[:]))
