#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

import argparse
import json
import numpy as np
import pickle
import sys

from untangled.cmdargs import AutoBool, FileExists, FileAbsent

json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

parser = argparse.ArgumentParser(description='Dump JSON representation of model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--out_file', default=None, action=FileAbsent, help='Output JSON file to this file location')
parser.add_argument('--params', default=True, action=AutoBool, help='Output parameters as well as model structure')

parser.add_argument('model', action=FileExists, help='Model file to read from')

#
# Some numpy types are not serializable to JSON out-of-the-box in Python3 -- need coersion. See
# http://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
# If numpy type is used to construct the layer, it will infect the pickle.
#


class CustomEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super(CustomEncoder, self).default(obj)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.model, 'rb') as fh:
        if sys.version_info.major == 3:
            model = pickle.load(fh, encoding='latin1')
        else:
            model = pickle.load(fh)

    json_out = model.json(args.params)

    if args.out_file is not None:
        mode = 'w' if sys.version_info.major == 3 else 'wb'
        with open(args.out_file, mode) as f:
            print("Writing to file: ", args.out_file)
            json.dump(json_out, f, indent=4, cls=CustomEncoder)
    else:
        print(json.dumps(json_out, indent=4, cls=CustomEncoder))
