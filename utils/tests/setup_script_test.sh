#!/bin/bash

log() {
  COLOR_DEFAULT='\033[0m'
  COLOR_RED='\033[91m'
  echo -e "${COLOR_RED}$1${COLOR_DEFAULT}"
}

if [ -z "$1" ]
then
	log "Please specify name of latest TF release Docker image, like: bash setup_script_test.sh tf_release_image torch_release_image ort_release_image"
	exit
fi

if [ -z "$2" ]
then
        log "Please specify name of latest Torch release Docker image, like: bash setup_script_test.sh tf_release_image torch_release_image ort_release_image"
        exit
fi

if [ -z "$3" ]
then
        log "Please specify name of latest ORT release Docker image, like: bash setup_script_test.sh tf_release_image torch_release_image ort_release_image"
        exit
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker stop aml_setup_test
sleep 5

log "Starting test ..."

if bash $SCRIPT_DIR/setup_test_utils/test.sh $1 $2 $3; then
    log "\n TEST FINISHED: SUCCESS"
else
    log "\n TEST FINISHED: FAIL"
fi

docker stop aml_setup_test
