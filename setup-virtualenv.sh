#!/bin/bash -ex

export ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source environment

# for fromscratch builds plow through ${SLOIKA_VIRTUALENV_DIR} before running this script
mkdir -p ${SLOIKA_VIRTUALENV_DIR}
virtualenv ${SLOIKA_VIRTUALENV_DIR}

# need a version of pip that supports --trusted-host option
pip install pip --upgrade
