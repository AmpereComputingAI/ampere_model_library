#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

log() {
  COLOR_DEFAULT='\033[0m'
  COLOR_RED='\033[91m'
  echo -e "${COLOR_RED}$1${COLOR_DEFAULT}"
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker stop aml_setup_test
sleep 5

log "Starting test ..."

if bash $SCRIPT_DIR/setup_test_utils/test_x86.sh; then
  log "\n TEST FINISHED: SUCCESS"
else
  log "\n TEST FINISHED: FAIL"
fi

docker stop aml_setup_test
