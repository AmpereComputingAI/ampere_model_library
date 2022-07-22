#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

log() {
  COLOR_DEFAULT='\033[0m'
  COLOR_CYAN='\033[1;36m'
  echo -e "${COLOR_CYAN}$1${COLOR_DEFAULT}"
}

# if the script is run with Jenkins, then set the SCRIPT_DIR manually, otherwise use the old way (SET JENKINS TO 1)
if [[ -z "${JENKINS}" ]]; then
  # not run with JENKINS
  if [ -z ${SCRIPT_DIR+x} ]; then
  SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
  fi
else
  # run with JENKINS, need to set the SCRIPT_DIR in a different way, more manually
  SCRIPT_DIR=`pwd`
  SCRIPT_DIR=$SCRIPT_DIR/ampere_model_library
fi


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
#   LD_PRELOAD=`cat $SCRIPT_DIR/utils/setup/.ld_preload`
   # add this path manually, due to the fact that python doesn't find it. (temporary hack?)
#   LD_PRELOAD=$LD_PRELOAD:/root/miniforge3/envs/tensorflow/lib/python3.8/site-packages/skimage/_shared/../../scikit_image.libs/libgomp-d22c30c5.so.1.0.0
   LD_PRELOAD=/root/miniforge3/envs/tensorflow/lib/python3.8/site-packages/skimage/_shared/../../scikit_image.libs/libgomp-d22c30c5.so.1.0.0
   echo "LD_PRELOAD=$LD_PRELOAD"
fi
export PYTHONPATH=$SCRIPT_DIR
echo "PYTHONPATH=$PYTHONPATH"
log "done.\n"
