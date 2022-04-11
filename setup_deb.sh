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

log "Installing systems dependencies ..."
sleep 1
apt-get update -y
apt-get install -y python-is-python3 python3-pip ffmpeg libsm6 libxext6
log "done.\n"

log "Installing python dependencies ..."
sleep 1
pip3 install -r $SCRIPT_DIR/requirements.txt
log "done.\n"

touch $SCRIPT_DIR/.setup_completed
