#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

set -eo pipefail

log() {
  COLOR_DEFAULT='\033[0m'
  COLOR_RED='\033[91m'
  echo -e "${COLOR_RED}$1${COLOR_DEFAULT}"
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

image_name="ubuntu:20.04"
log "\nStarting $image_name test ...\n"
docker run --privileged --name aml_setup_test --rm -d -v $SCRIPT_DIR/../../../:/aml -it $image_name
docker exec -i aml_setup_test bash -c 'export TZ=Europe/Warsaw; export DEBIAN_FRONTEND=noninteractive; bash /aml/setup_deb.sh; source /aml/set_env_variables.sh; python3 /aml/utils/tests/setup_test_utils/attempt_imports.py; exit $?'
docker stop aml_setup_test
sleep 5

image_name="ubuntu:21.10"
log "\nStarting $image_name test ...\n"
docker run --privileged --name aml_setup_test --rm -d -v $SCRIPT_DIR/../../../:/aml -it $image_name
docker exec -i aml_setup_test bash -c 'export TZ=Europe/Warsaw; export DEBIAN_FRONTEND=noninteractive; bash /aml/setup_deb.sh; source /aml/set_env_variables.sh; python3 /aml/utils/tests/setup_test_utils/attempt_imports.py; exit $?'
docker stop aml_setup_test
sleep 5

image_name="debian:11.0"
log "\nStarting $image_name test ...\n"
docker run --privileged --name aml_setup_test --rm -d -v $SCRIPT_DIR/../../../:/aml -it $image_name
docker exec -i aml_setup_test bash -c 'export TZ=Europe/Warsaw; export DEBIAN_FRONTEND=noninteractive; bash /aml/setup_deb.sh; source /aml/set_env_variables.sh; python3 /aml/utils/tests/setup_test_utils/attempt_imports.py; exit $?'
docker stop aml_setup_test
sleep 5

image_name="debian:11.3"
log "\nStarting $image_name test ...\n"
docker run --privileged --name aml_setup_test --rm -d -v $SCRIPT_DIR/../../../:/aml -it $image_name
docker exec -i aml_setup_test bash -c 'export TZ=Europe/Warsaw; export DEBIAN_FRONTEND=noninteractive; bash /aml/setup_deb.sh; source /aml/set_env_variables.sh; python3 /aml/utils/tests/setup_test_utils/attempt_imports.py; exit $?'
docker stop aml_setup_test
sleep 5

image_name=$1
log "\nStarting $image_name test ...\n"
docker run --privileged --name aml_setup_test --rm -d -v $SCRIPT_DIR/../../../:/aml -it $image_name
docker exec -i aml_setup_test bash -c 'export TZ=Europe/Warsaw; export DEBIAN_FRONTEND=noninteractive; bash /aml/setup_deb.sh; source /aml/set_env_variables.sh; python3 /aml/utils/tests/setup_test_utils/attempt_imports.py; exit $?'
docker stop aml_setup_test
sleep 5

image_name=$2
log "\nStarting $image_name test ...\n"
docker run --privileged --name aml_setup_test --rm -d -v $SCRIPT_DIR/../../../:/aml -it $image_name
docker exec -i aml_setup_test bash -c 'export TZ=Europe/Warsaw; export DEBIAN_FRONTEND=noninteractive; bash /aml/setup_deb.sh; source /aml/set_env_variables.sh; python3 /aml/utils/tests/setup_test_utils/attempt_imports.py; exit $?'
docker stop aml_setup_test
sleep 5

image_name=$3
log "\nStarting $image_name test ...\n"
docker run --privileged --name aml_setup_test --rm -d -v $SCRIPT_DIR/../../../:/aml -it $image_name
docker exec -i aml_setup_test bash -c 'export TZ=Europe/Warsaw; export DEBIAN_FRONTEND=noninteractive; bash /aml/setup_deb.sh; source /aml/set_env_variables.sh; python3 /aml/utils/tests/setup_test_utils/attempt_imports.py; exit $?'
docker stop aml_setup_test
