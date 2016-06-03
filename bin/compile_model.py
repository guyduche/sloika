#!/usr/bin/env python
import argparse
from six.moves import cPickle
import sys

from sloika import layers
from tang.util.cmdargs import FileExist

parser = argparse.ArgumentParser(
    description='Compile a pickled model file for basecalling',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', action=FileExist, help='Input model pickle')
parser.add_argument('output', help='output file for compiled pickle')

if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    args = parser.parse_args()

    with open(args.input, 'r') as fh:
        network = cPickle.load(fh)
    if not isinstance(network, layers.Layer):
        sys.stderr.write("Model file is not a network description. Is it already compiled?\n")
        exit(1)

    compiled_network = network.compile()
    with open(args.output, 'wb') as fh:
        cPickle.dump(compiled_network, fh, protocol=cPickle.HIGHEST_PROTOCOL)
