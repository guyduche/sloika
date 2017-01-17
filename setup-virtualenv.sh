#!/bin/bash -ex

source environment

# for fromscratch builds plow through ${SLOIKA_VIRTUALENV_DIR} before running this script
mkdir -p ${SLOIKA_VIRTUALENV_DIR}
virtualenv ${SLOIKA_VIRTUALENV_DIR}

source ${SLOIKA_VIRTUALENV_DIR}/bin/activate

# need a version of pip that supports --trusted-host option
pip install pip --upgrade

# install prerequisites of setup.py first
pip install \
    -r scripts/requirements.txt \
    --trusted-host pypi.oxfordnanolabs.local \
    --index-url https://pypi.oxfordnanolabs.local/simple/

#
# TODO(semen): take care when installing non-sloika deps into environment that is used in acctests
#
pip install \
    -r test/unit/requirements.txt \
    -r test/acceptance/requirements.txt \
    -r requirements.txt \
    --trusted-host pypi.oxfordnanolabs.local \
    --index-url https://pypi.oxfordnanolabs.local/simple/
