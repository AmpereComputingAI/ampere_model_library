#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

set -eo pipefail

log() {
    COLOR_DEFAULT='\033[0m'
    COLOR_CYAN='\033[1;36m'
    echo -e "${COLOR_CYAN}$1${COLOR_DEFAULT}"
}

ARCH=$(uname -m)

if [ -z ${SCRIPT_DIR+x} ]; then
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
fi

if [ ! -f "$SCRIPT_DIR/speech_recognition/whisper/whisper/README.md" ]; then
    log "Please pull submodules first: git submodule update --init --recursive"
    exit 1
fi

if [ "$FORCE_INSTALL" != "1" ]; then
    log "Checking for aarch64 system ..."
    sleep 1
    if [ "${ARCH}" != "aarch64" ]; then
        log "\nDetected $ARCH-based system while aarch64 one is expected. Quitting."
        exit 1
    fi
    log "done.\n"

    log "Checking for RHEL ..."
    sleep 1
    if [ -f "/etc/redhat-release" ]; then
        rhel_version=$(</etc/redhat-release)
        log "Detected $rhel_version. Be advised that this script supports RHEL>=9.4."
        sleep 3
    else
        log "\nRed Hat Linux has not been detected! Quitting."
        exit 1
    fi
    log "done.\n"
fi

log "Installing system dependencies ..."
sleep 1
yum install epel-release || :
yum groupinstall -y 'Development Tools'
yum install -y python3 python3-devel python3-pip libSM libXext wget git unzip numactl hdf5-devel cmake gcc-c++ 
git clone -b n4.3.7 https://github.com/FFmpeg/FFmpeg.git && cd FFmpeg && ./configure && make -j && make install && cd .. && rm -r FFmpeg
log "done.\n"

log "Setup LD_PRELOAD ..."
sleep 1
if [ "${ARCH}" = "aarch64" ]; then
    python3 "$SCRIPT_DIR"/utils/setup/gen_ld_preload.py
    LD_PRELOAD=$(cat "$SCRIPT_DIR"/utils/setup/.ld_preload)
    echo "LD_PRELOAD=$LD_PRELOAD"
fi
export LD_PRELOAD=$LD_PRELOAD
log "done.\n"

log "Installing python dependencies ..."
sleep 1

# get almost all python deps
pip3 install --break-system-packages --upgrade -r "$(dirname "$0")/requirements.txt" ||
    pip3 install --upgrade -r "$(dirname "$0")/requirements.txt"

yum install -y autoconf automake alsa-lib-devel pkg-config

# if [ "$(PYTHONPATH=$SCRIPT_DIR python3 -c 'from cpuinfo import get_cpu_info; from benchmark import which_ampere_cpu; cpu = which_ampere_cpu(get_cpu_info()["flags"], 1); print("AmpereOne" in cpu)')" == "True" ]; then
#     # Only on AmpereOne family
#     pip3 install --break-system-packages --upgrade -r "$(dirname "$0")/requirements-ampereone.txt" ||
#         pip3 install --upgrade -r "$(dirname "$0")/requirements-ampereone.txt"
# fi

ARCH=$ARCH python3 "$SCRIPT_DIR"/utils/setup/install_frameworks.py

if [ "$(python3 -c 'import torch; print(torch.cuda.is_available())')" == "True" ]; then
    # Torchvision version has to match PyTorch version following this table:
    # https://github.com/pytorch/vision?tab=readme-ov-file#installation
    pip3 install --no-build-isolation git+https://github.com/pytorch/vision.git@v0.16.1
fi
log "done.\n"

if [ -f "/etc/machine-id" ]; then
    cat /etc/machine-id >"$SCRIPT_DIR"/.setup_completed
else
    touch "$SCRIPT_DIR"/.setup_completed
fi
log "Setup completed. Please run: source $SCRIPT_DIR/set_env_variables.sh"
