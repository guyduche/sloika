#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import argparse
import pickle
import json
from untangled.cmdargs import AutoBool, FileExists, FileAbsent

json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

parser = argparse.ArgumentParser(description='Dump JSON representation of model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--out_file', default=None, action=FileAbsent, help='Output JSON file to this file location')
parser.add_argument('--params', default=True, action=AutoBool, help='Output parameters as well as model structure')

parser.add_argument('model', action=FileExists, help='Model file to read from')

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.model, 'rb') as fh:
        model = pickle.load(fh)

    json_out = model.json(args.params)

    if args.out_file is not None:
        with open(args.out_file, 'wb') as f:
            print("Writing to file: ", args.out_file)
            json.dump(json_out, f, indent=4)
    else:
        print(json.dumps(json_out, indent=4))
