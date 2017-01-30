#!/bin/bash

export SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source ${SCRIPTS_DIR}/version-env.sh

echo -n ${SLOIKA_VERSION}
