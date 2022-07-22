#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

log() {
  COLOR_DEFAULT='\033[0m'
  COLOR_CYAN='\033[1;36m'
  echo -e "${COLOR_CYAN}$1${COLOR_DEFAULT}"
}
echo "pwd: `pwd`"
if [ -z ${SCRIPT_DIR+x} ]; then
#  SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
  SCRIPT_DIR=`pwd`
fi

echo $SCRIPT_DIR
SCRIPT_DIR=$SCRIPT_DIR/ampere_model_library
echo $SCRIPT_DIR


log "Checking if setup has been completed ..."
sleep 1
if ! [ -f "$SCRIPT_DIR/.setup_completed" ]; then
   log "\nPlease complete setup first by running: 'bash setup_deb.sh'! Quitting."
   exit 1
fi
log "done.\n"

log "Checking if script is sourced ..."
sleep 1
(return 0 2>/dev/null) && SOURCED=1 || SOURCED=0
if [ $SOURCED != 1 ]; then
   log "\nPlease source the script: 'source set_env_variables.sh'. Quitting."
   exit 1
fi
log "done.\n"

log "Setting environment variables ..."
sleep 1
ARCH=$( uname -m )
if [ "${ARCH}" = "aarch64" ]; then
   python3 $SCRIPT_DIR/utils/setup/gen_ld_preload.py
   LD_PRELOAD=$(<$SCRIPT_DIR/utils/setup/.ld_preload)
   export LD_PRELOAD=$LD_PRELOAD
   echo "LD_PRELOAD=$LD_PRELOAD"
fi
export PYTHONPATH=$SCRIPT_DIR
echo "PYTHONPATH=$PYTHONPATH"
log "done.\n"
