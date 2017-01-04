#!/bin/bash -ex

export ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source environment


mkdir -p ${SLOIKA_SITE_PACKAGES}

sudo pip install -r requirements.txt --trusted-host pypi.oxfordnanolabs.local \
	-i https://pypi.oxfordnanolabs.local/simple/ 

python setup.py develop --prefix ${SLOIKA_DEV_DIR}
