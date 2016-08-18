#!/usr/bin/env python
import argparse
import imp
import os
import sys

import theano as th
import theano.tensor as T

from untangled.cmdargs import (display_version_and_exit, FileExists,
                               Positive)

from sloika import __version__

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Train a simple transducer neural network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--kmer', default=5, metavar='length', type=Positive(int),
    help='Length of kmer')
parser.add_argument('--sd', default=0.5, metavar='value', type=Positive(float),
    help='Standard deviation to initialise with')
parser.add_argument('--window', default=3, type=Positive(int), metavar='length',
    help='Window length for input features')
parser.add_argument('--version', nargs=0, action=display_version_and_exit, metavar=__version__,
    help='Display version information.')
parser.add_argument('model', metavar='file.py', action=FileExists,
    help='File to read python model description from')


def wrap_network(network):
    x = T.tensor3()
    labels = T.imatrix()
    post = network.run(x)
    loss = T.mean(th.map(T.nnet.categorical_crossentropy, sequences=[post, labels])[0])
    ncorrect = T.sum(T.eq(T.argmax(post,  axis=2), labels))

    fg = th.function([x, labels], [loss, ncorrect])
    return fg


if __name__ == '__main__':
    args = parser.parse_args()

    #  Set some Theano options
    th.config.optimizer = 'fast_compile'
    th.config.warn_float64 = 'warn'

    try:
        netmodule = imp.load_source('netmodule', args.model)
        network = netmodule.network(winlen=args.window, klen=args.kmer, sd=args.sd)
        fg = wrap_network(network)
    except:
        sys.stderr.write('Compilation of model {} failed\n'.format(args.model))
        raise
        exit(1)

    nparam = sum([p.get_value().size for p in network.params()])
    sys.stderr.write('Compilation of model {} succeeded\n'.format(os.path.basename(args.model)))
    sys.stderr.write('nparam = {}\n'.format(nparam))
    exit(0)
