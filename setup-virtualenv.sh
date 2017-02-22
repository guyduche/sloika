#!/bin/bash -ex

source environment

# for fromscratch builds plow through ${DEV_VIRTUALENV_DIR} before running this script
mkdir -p ${DEV_VIRTUALENV_DIR}
virtualenv ${DEV_VIRTUALENV_DIR}

source ${DEV_VIRTUALENV_DIR}/bin/activate

# need a version of pip that supports --trusted-host option
pip install pip --upgrade

# install prerequisites of setup.py first
pip install \
    -r scripts/requirements.txt \
    --trusted-host pypi.oxfordnanolabs.local \
    --index-url https://pypi.oxfordnanolabs.local/simple/

# install Sloika's dependencies
pip install \
    -r requirements.txt \
    --trusted-host pypi.oxfordnanolabs.local \
    --index-url https://pypi.oxfordnanolabs.local/simple/
