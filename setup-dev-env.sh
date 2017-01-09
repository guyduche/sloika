#!/bin/bash -ex

source environment

source ${SLOIKA_VIRTUALENV_DIR}/bin/activate

python setup.py develop
