#!/bin/bash

set -euo pipefail

log() {
  COLOR_DEFAULT='\033[0m'
  COLOR_CYAN='\033[1;36m'
  echo -e "${COLOR_CYAN}$1${COLOR_DEFAULT}"
}

log "Checking for Debian based Linux ..."
sleep 1
if ! [ -f "/etc/debian_version" ]; then
   log "\nDebian-based Linux has not been detected! Quitting."
   exit 1
fi
log "done.\n"

log "Checking for aarch64 system ..."
sleep 1
ARCH=$( uname -m )
if [ ${ARCH} != "aarch64" ]; then
   log "\nDetected $ARCH-based system while aarch64 one is expected. Quitting."
   exit 1
fi
log "done.\n"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

log "Installing system dependencies ..."
sleep 1
apt-get update -y
apt-get install -y python3.8 python3-pip ffmpeg libsm6 libxext6 wget
log "done.\n"

log "Installing python dependencies ..."
sleep 1
#PYTHON3_VERSION=$( pip3 --version )
#if echo "$PYTHON3_VERSION" | grep -q "(python 3.8)"; then
    #wget https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/SimpleITK_aarch64-2.1.1-cp38-cp38-linux_aarch64.whl
#else
#   log "\nCompatible Python versions are: 3.8 and 3.6. Version detected: $PYTHON3_VERSION. Quitting."
#   exit 1
#fi
python3.8 -m pip install -r $SCRIPT_DIR/requirements.txt
log "done.\n"

touch $SCRIPT_DIR/.setup_completed
log "Setup completed."
