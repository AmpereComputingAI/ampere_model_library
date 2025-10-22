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
apt-get install -y build-essential ffmpeg libsm6 libxext6 wget git unzip numactl libhdf5-dev cmake
if ! python3 -c ""; then
    apt-get update -y
    apt-get install -y python3 python3-pip
fi
if ! pip3 --version; then
    apt-get install -y python3-pip
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
PYTHON_DEV_SEARCH=$(apt-cache search --names-only "python${PYTHON_VERSION}-dev")
if [[ -n "$PYTHON_DEV_SEARCH" ]]; then
    apt-get install -y "python${PYTHON_VERSION}-dev"
fi
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

ARCH=$ARCH python3 "$SCRIPT_DIR"/utils/setup/install_frameworks.py

# get almost all python deps
PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --ignore-installed --upgrade pip
python3 -m pip install --break-system-packages -r "$(dirname "$0")/requirements.txt" ||
    python3 -m pip3 install -r "$(dirname "$0")/requirements.txt"

apt install -y autoconf autogen automake build-essential libasound2-dev \
    libflac-dev libogg-dev libtool libvorbis-dev libopus-dev libmp3lame-dev \
    libmpg123-dev pkg-config ffmpeg
apt remove -y libsndfile1
git clone -b 1.2.2 https://github.com/libsndfile/libsndfile.git && cd libsndfile/ && autoreconf -vif && ./configure --enable-werror && make -j && make install && ldconfig && cd .. && rm -rf libsndfile

# if [ "$(PYTHONPATH=$SCRIPT_DIR python3 -c 'from cpuinfo import get_cpu_info; from benchmark import which_ampere_cpu; cpu = which_ampere_cpu(get_cpu_info()["flags"], 1); print("AmpereOne" in cpu)')" == "True" ]; then
#     # Only on AmpereOne family
#     pip3 install --break-system-packages -r "$(dirname "$0")/requirements-ampereone.txt" ||
#         pip3 install -r "$(dirname "$0")/requirements-ampereone.txt"
# fi

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
