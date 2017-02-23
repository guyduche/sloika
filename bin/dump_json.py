#!/usr/bin/env python
import argparse
import cPickle
import json
from untangled.cmdargs import AutoBool, FileExists

json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

parser = argparse.ArgumentParser(description='Dump JSON representation of model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--params', default=True, action=AutoBool, help='Output parameters in file')
parser.add_argument('--out_file', default=False, help='[Opt] Location of output json file')
parser.add_argument('model', action=FileExists, help="Model file to read from")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.model, 'r') as fh:
        model = cPickle.load(fh)

    json_out = model.json(args.params)

    if args.out_file:
        with open(args.out_file, 'w') as f:
            print "Writing to file: ", args.out_file
            json.dump(json_out, f, indent=4)
    else:
        print json.dumps(json_out, indent=4)
