#!/usr/bin/env python
import argparse
import cPickle
import json
from untangled.cmdargs import AutoBool, FileExists
json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

parser = argparse.ArgumentParser(
    description = 'Dump JSON representation of model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--params', default=True, action=AutoBool,
    help='Output parameters in file')
parser.add_argument('model', action=FileExists, help="Model file to read from")


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.model, 'r') as fh:
        model = cPickle.load(fh)

    print json.dumps(model.json(args.params), indent=4)
