#!/bin/bash -ex

source environment

if [ -z ${PY3+x} ]; then
    VIRTUALENV_DIR="${DEV_VIRTUALENV_DIR}"
    VIRTUALENV_CMD=virtualenv
else
    VIRTUALENV_DIR="${DEV_VIRTUALENV_DIR_PY3}"
    VIRTUALENV_CMD="virtualenv -p python3"
fi

# for fromscratch builds plow through ${VIRTUALENV_DIR} before running this script
mkdir -p ${VIRTUALENV_DIR}
${VIRTUALENV_CMD} ${VIRTUALENV_DIR}

source ${VIRTUALENV_DIR}/bin/activate

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
