#!/bin/bash -ex

source environment

source ${DEV_VIRTUALENV_DIR}/bin/activate

# need a version of pip that supports --trusted-host option
pip install pip --upgrade

pip install \
    -r setup-dev-env.txt \
    --trusted-host pypi.oxfordnanolabs.local \
    --index-url https://pypi.oxfordnanolabs.local/simple/

python setup.py develop
