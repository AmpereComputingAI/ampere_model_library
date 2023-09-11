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
apt-get install -y python3 python3-pip build-essential ffmpeg libsm6 libxext6 wget
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
PYTHON_DEV_SEARCH=$(apt-cache search --names-only "python${PYTHON_VERSION}-dev")
if [[ -n "$PYTHON_DEV_SEARCH"  ]]; then
	apt-get -y install "python${PYTHON_VERSION}-dev"
fi
log "done.\n"

log "Setup LD_PRELOAD ..."
sleep 1
if [ "${ARCH}" = "aarch64" ]; then
   python3 $SCRIPT_DIR/utils/setup/gen_ld_preload.py
   LD_PRELOAD=`cat $SCRIPT_DIR/utils/setup/.ld_preload`
   echo "LD_PRELOAD=$LD_PRELOAD"
fi
export LD_PRELOAD=$LD_PRELOAD
log "done.\n"

log "Installing python dependencies ..."
sleep 1
# direct dependencies
pip3 install --no-deps --upgrade \
   SimpleITK==2.2.1 \
   batchgenerators==0.21 \
   medpy==0.4.0 \
   nibabel==3.2.2 \
   "numpy<1.24.0" \
   opencv-python==4.5.5.64 \
   pandas==1.4.2 \
   pycocotools==2.0.6 \
   scikit-build==0.14.1 \
   scipy==1.8.0 \
   tifffile==2023.1.23.1 \
   transformers==4.30.1 \
   tqdm==4.64.0 \
   sacrebleu==2.3.1 \
   sentencepiece==0.1.97 \
   tiktoken==0.3.3 \
   ultralytics==8.0.75 \
   evaluate==0.4.0 \
   datasets==2.13.1 \
   soundfile==0.12.1 \
   librosa==0.10.0.post2 \
   numba==0.57.1

# dependencies of dependencies
pip3 install --no-deps --upgrade \
   cycler==0.11.0 \
   filelock==3.6.0 \
   future==0.18.2 \
   huggingface-hub==0.13.4 \
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
   tokenizers==0.13.3 \
   tabulate==0.9.0 \
   regex==2022.3.15 \
   portalocker==2.6.0 \
   lxml==4.9.2 \
   colorama==0.4.6 \
   thop>=0.1.1 \
   psutil==5.9.5 \
   multiprocess==0.70.14 \
   pyarrow==12.0.1 \
   dill==0.3.6 \
   aiohttp==3.8.4 \
   multidict==6.0.4 \
   yarl==1.9.2 \
   async_timeout==4.0.2 \
   aiosignal==1.3.1 \
   frozenlist==1.3.3 \
   xxhash==3.2.0 \
   lazy_loader==0.2 \
   audioread==3.0.0 \
   soxr==0.3.5 \
   msgpack==1.0.5 \
   jiwer==3.0.2 \
   click==8.1.3 \
   rapidfuzz==2.13.7 \
   llvmlite==0.40.1 \
   decorator==5.1.1 \
   fsspec==2023.6.0 \
   unicode==2.9 \
   unidecode==1.3.6 \
   inflect==6.0.4 \
   pydantic==1.10.9 \
   cffi==1.15.1 \
   attrs==23.1.0 \
   safetensors>=0.3.1

ARCH=$ARCH python3 "$SCRIPT_DIR"/utils/setup/install_frameworks.py
log "done.\n"

touch "$SCRIPT_DIR"/.setup_completed
log "Setup completed. Please run: source $SCRIPT_DIR/set_env_variables.sh"
