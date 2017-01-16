#!/bin/bash -ex

source environment

# for fromscratch builds plow through ${SLOIKA_VIRTUALENV_DIR} before running this script
mkdir -p ${SLOIKA_VIRTUALENV_DIR}
virtualenv ${SLOIKA_VIRTUALENV_DIR}

source ${SLOIKA_VIRTUALENV_DIR}/bin/activate

# need a version of pip that supports --trusted-host option
pip install pip --upgrade

pip install \
    -r scripts/requirements.txt \
    -r test/unit/requirements.txt \
    -r test/acceptance/requirements.txt \
    -r requirements.txt \
    --trusted-host pypi.oxfordnanolabs.local \
    --index-url https://pypi.oxfordnanolabs.local/simple/
