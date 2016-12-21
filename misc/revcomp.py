#!/usr/bin/env python
import argparse
import sys
from Bio import SeqIO
from untangled.cmdargs import AutoBool

parser = argparse.ArgumentParser('Reverse complements fasta or fastq files')
parser.add_argument('--complement', default=True, action=AutoBool, help='Complement input')
parser.add_argument('--fastq', default=False, action=AutoBool, help='Input is fastq')
parser.add_argument('--reverse', default=True, action=AutoBool, help='Reverse input')
parser.add_argument('fasta', metavar='fasta', nargs='+', help='Files to read from')

if __name__ == '__main__':
    args = parser.parse_args()
    seqtype = 'fastq' if args.fastq else 'fasta'
    rc_flags = {'id': True,
                'name' : True,
                'description' : True,
                'features' : False,
                'annotations' : False,
                'letter_annotations' : True,
                'dbxrefs' : False}

    for fn in args.fasta:
        for seq in SeqIO.parse(fn, seqtype):
            do_complement = args.complement
            if args.reverse:
                # Contrived ordering here so biopython deals with reversing seq and quality
                seq = seq.reverse_complement(**rc_flags)
                do_complement = not do_complement
            if do_complement:
                seq.seq = seq.seq.complement()

            SeqIO.write(seq, sys.stdout, seqtype)
