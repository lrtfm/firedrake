#!/bin/bash

set -e

TOOL=${TOOL:-docker}

if [[ "$TOOL" == "docker" ]]; then
    BUILD_ARGS="${BUILD_ARGS:---network=host}"
elif [[ "$TOOL" == "podman" ]]; then
    BUILD_ARGS="${BUILD_ARGS} --format docker"  # Use Docker format for Podman
else
    echo "Error: TOOL must be set to either 'docker' or 'podman'."
    exit 1
fi

${TOOL} build ${BUILD_ARGS} -t lrtfm/firedrake-env -f Dockerfile.env .
${TOOL} push lrtfm/firedrake-env

${TOOL} build ${BUILD_ARGS} -t lrtfm/firedrake-vanilla -f Dockerfile.vanilla .
${TOOL} push lrtfm/firedrake-vanilla

${TOOL} build ${BUILD_ARGS} -t lrtfm/firedrake-complex -f Dockerfile.complex .
${TOOL} push lrtfm/firedrake-complex

${TOOL} build ${BUILD_ARGS} -t lrtfm/firedrake -f Dockerfile.firedrake .
${TOOL} push lrtfm/firedrake
