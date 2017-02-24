#!/bin/bash -eu

export SLOIKA_VERSION_MAJOR=1
export SLOIKA_VERSION_MINOR=1

if hash git 2>/dev/null; then
    export SLOIKA_VERSION_PATCH=$(git rev-list --count HEAD)
else
    export SLOIKA_VERSION_PATCH=0
fi

if [ "${CI_BUILD_REF_NAME:-dev}" == "master" ]; then
    DEV=""
else
    DEV="dev"
fi

export SLOIKA_VERSION=${SLOIKA_VERSION_MAJOR}.${SLOIKA_VERSION_MINOR}.${DEV}${SLOIKA_VERSION_PATCH}
