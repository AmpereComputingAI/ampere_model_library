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
apt-get install -y python3-pip ffmpeg libsm6 libxext6 wget
log "done.\n"

log "Installing python dependencies ..."
sleep 1
PYTHON3_VERSION=$( pip3 --version )
if ! echo "$PYTHON3_VERSION" | grep -q "(python 3.8)"; then
   log "\nThis script requires python 3.8! Quitting."
   exit 1
fi
#python3.8 -m pip install -r $SCRIPT_DIR/requirements.txt
pip3 install --no-deps --upgrade https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/SimpleITK_aarch64-2.1.1-cp38-cp38-linux_aarch64.whl
#pip3 install -r $SCRIPT_DIR/requirements.txt
pip3 install --no-deps --upgrade medpy==0.4.0 batchgenerators==0.21 medpy==0.4.0 nibabel==3.2.2 numpy==1.22.3 opencv-python==4.5.5.64 pandas==1.4.2 pycocotools==2.0.4 scikit-build==0.14.1 scipy==1.8.0 torchvision==0.10.0 transformers==4.18.0 tqdm==4.64.0
pip3 install --no-deps --upgrade cycler==0.11.0 filelock==3.6.0 future==0.18.2 huggingface-hub==0.5.1 joblib==1.1.0 kiwisolver==1.4.2 matplotlib==3.5.1 nnunet==1.7.0 packaging==21.3 Pillow==9.1.0 pyparsing==3.0.8 python-dateutil==2.8.2 pytz==2022.1 pyyaml==6.0 regex==2022.3.15 sacremoses==0.0.49 scikit-image==0.19.2 scikit-learn==1.0.2 threadpoolctl==3.1.0 tokenizers==0.12.1
python3 install_frameworks.py
log "done.\n"

touch $SCRIPT_DIR/.setup_completed
log "Setup completed."
