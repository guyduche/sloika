#!/bin/bash -ex

source environment

VIRTUALENV_DIR="${DEV_VIRTUALENV_DIR}"

source ${VIRTUALENV_DIR}/bin/activate

# need a version of pip that supports --trusted-host option
pip install pip --upgrade

pip install \
    -r misc/requirements.txt \
    --trusted-host pypi.oxfordnanolabs.local \
    --index-url https://pypi.oxfordnanolabs.local/simple/

pip install \
    -r setup-dev-env.txt \
    --trusted-host pypi.oxfordnanolabs.local \
    --index-url https://pypi.oxfordnanolabs.local/simple/

python setup.py develop
