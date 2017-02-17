#!/bin/bash -eu

export SLOIKA_VERSION_MAJOR=1
export SLOIKA_VERSION_MINOR=1

if hash git 2>/dev/null; then
    export SLOIKA_VERSION_PATCH=$(git rev-list --count HEAD)
else
    export SLOIKA_VERSION_PATCH=0
fi

if [ -z ${CI+x} ]; then
    DEV="dev"
else
    DEV=""
fi

export SLOIKA_VERSION=${SLOIKA_VERSION_MAJOR}.${SLOIKA_VERSION_MINOR}.${DEV}${SLOIKA_VERSION_PATCH}
