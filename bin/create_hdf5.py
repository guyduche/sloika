#!/usr/bin/env python
from __future__ import print_function

import sys

from sloika.tools.create_hdf5 import create_hdf5_main
from sloika.tools.create_hdf5_with_remap import create_hdf5_with_remap_main


def main(argv):

    scripts = {
        'orig': create_hdf5_main,
        'remap': create_hdf5_with_remap_main,
    }

    if 1 == len(argv):
        print("Available commands:")
        print('\t'+'\n\t'.join(sorted(scripts.keys())))
    else:
        command_name = argv[1]
        command_arguments = argv[1:]
        command_function = scripts[command_name]

        try:
            return command_function(command_arguments)
        except:
            print('Exception when running {!r}'.format(command_name))
            raise


if __name__ == '__main__':
    sys.exit(main(sys.argv[:]))
