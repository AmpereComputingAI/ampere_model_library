#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

set -eo pipefail

log() {
  COLOR_DEFAULT='\033[0m'
  COLOR_CYAN='\033[1;36m'
  echo -e "${COLOR_CYAN}$1${COLOR_DEFAULT}"
}

ARCH=$( uname -m )

if [ -z ${SCRIPT_DIR+x} ]; then
  SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
fi

if [ "$FORCE_INSTALL" != "1" ]; then
   log "Checking for aarch64 system ..."
   sleep 1
   if [ "${ARCH}" != "aarch64" ]; then
      log "\nDetected $ARCH-based system while aarch64 one is expected. Quitting."
      exit 1
   fi
   log "done.\n"
   
   log "Checking for Debian based Linux ..."
   sleep 1
   if [ -f "/etc/debian_version" ]; then
      debian_version=$(</etc/debian_version)
      log "Detected Debian $debian_version. Be advised that this script supports Debian >=11.0."
      sleep 3
   else
      log "\nDebian-based Linux has not been detected! Quitting."
      exit 1
   fi
   log "done.\n"
fi

log "Installing system dependencies ..."
sleep 1
apt-get update -y
apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6 wget
log "done.\n"

log "Installing python dependencies ..."
sleep 1
if [ "${ARCH}" == "aarch64" ]; then
   PYTHON3_VERSION=$( pip3 --version )
   if echo "$PYTHON3_VERSION" | grep -q "(python 3.8)"; then
      pip3 install --no-deps --upgrade \
         https://nexusai.amperecomputing.com/repository/pypi-public/packages/simpleitk-aarch64/2.1.1/SimpleITK_aarch64-2.1.1-cp38-cp38-linux_aarch64.whl
   elif echo "$PYTHON3_VERSION" | grep -q "(python 3.9)"; then
      pip3 install --no-deps --upgrade \
         https://nexusai.amperecomputing.com/repository/pypi-public/packages/simpleitk-aarch64/2.1.1/SimpleITK_aarch64-2.1.1-cp39-cp39-linux_aarch64.whl
   elif echo "$PYTHON3_VERSION" | grep -q "(python 3.10)"; then
      pip3 install --no-deps --upgrade \
         https://nexusai.amperecomputing.com/repository/pypi-public/packages/simpleitk-aarch64/2.1.1/SimpleITK_aarch64_precompiled-2.1.1-cp310-cp310-linux_aarch64.whl
   else
      log "\nThis script requires python >=3.8! Quitting."
      exit 1
   fi
else
   pip3 install --no-deps --upgrade SimpleITK==2.1.1
fi
# direct dependencies
pip3 install --no-deps --upgrade \
   batchgenerators==0.21 \
   medpy==0.4.0 \
   nibabel==3.2.2 \
   "numpy>=1.22.3" \
   opencv-python==4.5.5.64 \
   pandas==1.4.2 \
   pycocotools==2.0.4 \
   scikit-build==0.14.1 \
   scipy==1.8.0 \
   transformers==4.18.0 \
   tqdm==4.64.0
# dependencies of dependencies
pip3 install --no-deps --upgrade \
   cycler==0.11.0 \
   filelock==3.6.0 \
   future==0.18.2 \
   huggingface-hub==0.5.1 \
   joblib==1.1.0 \
   kiwisolver==1.4.2 \
   matplotlib==3.5.1 \
   nnunet==1.7.0 \
   packaging==21.3 \
   Pillow==9.1.0 \
   pyparsing==3.0.8 \
   python-dateutil==2.8.2 \
   pytz==2022.1 \
   pyyaml==6.0 \
   regex==2022.3.15 \
   sacremoses==0.0.49 \
   scikit-image==0.19.2 \
   scikit-learn==1.0.2 \
   threadpoolctl==3.1.0 \
   tokenizers==0.12.1 \
   absl-py \
   wrapt \
   opt_einsum \
   gast \
   astunparse \
   termcolor \
   keras_preprocessing \
   tensorflow==2.7.1
ARCH=$ARCH python3 "$SCRIPT_DIR"/utils/setup/install_frameworks.py
log "done.\n"

touch "$SCRIPT_DIR"/.setup_completed
log "Setup completed. Please run: source $SCRIPT_DIR/set_env_variables.sh"
