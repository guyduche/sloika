#!/usr/bin/env python
from __future__ import print_function
from Bio import SeqIO
from collections import Counter, OrderedDict
import math
import sys
from untangled.bio import reverse_complement, reverse_complement_kmers, seq_to_kmers

BASES = ['A', 'C', 'G', 'T']
kmers = [6, 7, 8, 9, 10]
RARE = 1000


all_counts = dict()
for k in kmers:
    all_counts[k] = Counter()


first = True
for f in sys.argv[1:]:
    with open(f, 'r') as fh:
        seqrec = SeqIO.read(fh, 'fasta')

    info = OrderedDict()
    info['genome'] = seqrec.id
    info['length'] = len(seqrec.seq)
    comp = [seqrec.seq.count(b) for b in BASES]
    info['pGC'] = 100.0 * (comp[1] + comp[2]) / len(seqrec.seq)

    for k in kmers:
        nkmer = len(BASES) ** k
        sk = Counter(seq_to_kmers(str(seqrec.seq), k))
        sk += Counter({k : v for k, v in zip(reverse_complement_kmers(sk.keys()), sk.values())})
        info['missK' + str(k)] = nkmer - len(sk)
        info['absrareK' + str(k)] = info['missK' + str(k)] + sum([i < RARE for i in sk.values()])
        lowcount = math.ceil((0.1 * len(seqrec.seq)) / nkmer)
        info['relrareK' + str(k)] = info['missK' + str(k)] + sum([i < lowcount for i in sk.values()])

        all_counts[k] += sk

    if first:
        print(','.join(info.keys()))
        first = False
    print(','.join(str(v) for v in info.values()))

print('\n#Summary kmers')
print('kmer\tmissing\tabsrare\trelrare\tlowcount')
for k in kmers:
    nkmer = len(BASES) ** k
    nkmer - len(sk)
    missing = nkmer - len(all_counts[k])
    absrare = missing + sum([i < RARE for i in all_counts[k].values()])
    lowcount = math.ceil((0.1 * sum([i for i in all_counts[k].values()])) / nkmer)
    relrare = missing + sum([i < lowcount for i in all_counts[k].values()])
    print(k, missing, absrare, relrare)
