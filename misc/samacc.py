#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import pysam


parser = argparse.ArgumentParser(
    description='Output match statistics from SAM',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--coverage', metavar='proportion', default=0.8,
    help='Minimum coverage')
parser.add_argument('sam')

STRAND = { 0 : '+',
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
            coverage = float(read.query_alignment_length) / read.query_length
            if coverage < args.coverage:
                continue
            bins = np.zeros(9, dtype='i4')
            for flag, count in read.cigartuples:
                bins[flag] += count

            alnlen = np.sum(bins[:3])
            mismatch = read.get_tag('NM')
            correct = alnlen - mismatch
            print(ref_name[read.reference_id], read.query_name,
                  STRAND[read.flag], bins[0], mismatch, bins[1], bins[2],
                  float(read.query_alignment_length) / read.query_length,
                  float(bins[0]) / alnlen,
                  float(correct) / alnlen)
