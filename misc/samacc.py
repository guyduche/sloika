#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
from past.utils import old_div
import argparse
import numpy as np
import pysam
from untangled.cmdargs import proportion


parser = argparse.ArgumentParser(
    description='Output match statistics from SAM',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--coverage', metavar='proportion', default=0.8, type=proportion,
                    help='Minimum coverage')
parser.add_argument('sam')

STRAND = {0 : '+',
          16 : '-'}

if __name__ == '__main__':
    args = parser.parse_args()

    print('name1', 'name2', 'strand', 'match', 'mismatch', 'insertion',
          'deletion', 'coverage', 'id', 'accuracy')
    with pysam.Samfile(args.sam, 'r') as sf:
        ref_name = sf.references
        for read in sf:
            if read.flag != 0 and read.flag != 16:
                continue

            coverage = old_div(float(read.qlen), read.rlen)
            if coverage < args.coverage:
                continue

            bins = np.zeros(9, dtype='i4')
            for flag, count in read.cigar:
                bins[flag] += count

            tags = dict(read.tags)
            alnlen = np.sum(bins[:3])
            mismatch = tags['NM']
            correct = alnlen - mismatch
            print(ref_name[read.reference_id], read.qname,
                  STRAND[read.flag], bins[0], mismatch, bins[1], bins[2],
                  coverage,
                  old_div(float(correct), float(bins[0])),
                  old_div(float(correct), alnlen))
