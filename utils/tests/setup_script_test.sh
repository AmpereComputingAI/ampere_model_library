#!/bin/bash

set -eo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run --name aml_setup_test --rm -d -v $SCRIPT_DIR/../../:/aml -it ubuntu:20.04

docker exec -i aml_setup_test bash -c '/aml/setup_deb.sh; source /aml/set_env_variables.sh; python3 /aml/utils/tests/attempt_imports.py; exit $?'
