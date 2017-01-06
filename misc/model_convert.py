#!/usr/bin/env python
import argparse
import cPickle
import sys
from untangled.cmdargs import Maybe, FileExists
from sloika.layers import Layer
from theano.tensor.sharedvar import TensorSharedVariable
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
import theano as th

parser = argparse.ArgumentParser('Converts pickled sloika model between CPU and GPU (CUDA) versions')
parser.add_argument('--target', default='swap', type=str,
    choices=('cpu', 'gpu', 'swap'), help='Target device (cpu, gpu or swap)')
parser.add_argument('model', action=FileExists,
    help='Pickled sloika model to convert')
parser.add_argument('output', help='Output file to write to')

def move_shared_recursive(obj, target, depth=0, max_depth=5):
    """Move Theano shared variables in object to target device

    Hackety hack hack! Wee!!

    :param obj: object to traverse (depth first) looking for shared variables
    :param target: 'cpu', 'gpu' or None meaning to swap between cpu and gpu
    :param depth: depth of recursion so far
    :param max_depth: maximum recursion depth within any Layer instance
    """
    names = dir(obj)
    for n in filter(lambda x: x[0] != '_', names):
        try:
            x = getattr(obj, n)
            if isinstance(x, TensorSharedVariable):
                t = target or 'gpu'
                setattr(obj, n, th.shared(x.get_value(), target=t))
            elif isinstance(x, CudaNdarraySharedVariable):
                t = target or 'cpu'
                setattr(obj, n, th.shared(x.get_value(), target=t))
            elif isinstance(x, Layer):
                move_shared_recursive(x, target, 0, max_depth=max_depth)
            elif isinstance(x, list):
                # This handles the layers attribute of layers.Serial and
                # layers.Parallel. Max_depth guards against runaway recursion.
                if depth < max_depth:
                    for item in x:
                        move_shared_recursive(item, target, depth + 1,
                                                        max_depth=max_depth)
        except AttributeError:
            # We hit an abstract method or property
            pass

if __name__ == '__main__':
    args = parser.parse_args()
    sys.stdout.write('Loading pickled model:  ' + args.model + '\n')
    with open(args.model, 'r') as fi:
        net = cPickle.load(fi)

    sys.stdout.write('Moving shared variables to target device\n')
    move_shared_recursive(net, args.target)

    sys.stdout.write('Writing new pickled model:  ' + args.output + '\n')
    with open(args.output, 'w') as fo:
        cPickle.dump(net, fo)
