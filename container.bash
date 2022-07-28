#!/usr/bin/env bash
# inputs
MODE=$1
VERSION=$2
GPU=$3
CMD=$4

# default paths
CURRENT_FOLDER="$(pwd)"
WANDB_KEY=06de2b089b5d98ee67dcf4fdffce3368e8bac2e4
USER=dkm
USER_ID=1003
USER_GROUP=dkm
USER_GROUP_ID=1003

if [[ $MODE == "build" ]]; then
  # build container
  docker build ./ -t $VERSION
  # here we use root permission instead of # -u 
  echo "=== MAKE refer"
  docker run -v $CURRENT_FOLDER/:/home/drigoni/repository/volta/ \
    -u root\
    --runtime=nvidia \
    $VERSION \
    bash -c 'cd ./tools/refer && make'
  echo "=== MAKE setup"
  docker run -v $CURRENT_FOLDER/:/home/drigoni/repository/volta/ \
    -u root\
    --runtime=nvidia \
    $VERSION \
    python setup.py develop
elif [[ $MODE == "exec" ]]; then
  echo "Remove previous container: "
  docker container rm ${VERSION}-${GPU//,}
  # execute container
  echo "Execute container:"
  docker run \
    -u ${USER}:${USER_GROUP} \
    --env CUDA_VISIBLE_DEVICES=${GPU} \
    --env WANDB_API_KEY=${WANDB_KEY}\
    --name ${VERSION}-${GPU//,} \
    --runtime=nvidia \
    --ipc=host \
    -it  \
    -v ${CURRENT_FOLDER}/:/home/drigoni/repository/volta/ \
    -v ${CURRENT_FOLDER}/data:/home/drigoni/repository/volta/data \
    -v ${CURRENT_FOLDER}/volta/datasets:/home/drigoni/repository/volta/volta/datasets \
    -v ${CURRENT_FOLDER}/refer/data:/home/drigoni/repository/volta/refer/data \
    -v ${CURRENT_FOLDER}/features_extraction:/home/drigoni/repository/volta/features_extraction \
    $VERSION \
    $CMD
    # '{"mode":0, "dataset":"flickr30k", "suffix":"kl1n0.4", "prefetch_factor":10, "num_workers":30, "align_loss":"kl", "regression_loss":"reg", "restore": null, "loss_weight_reg":1.0, "align_loss_kl_threshold":0.4}'
elif [[ $MODE == "interactive" ]]; then
  docker run -v $CURRENT_FOLDER/:/home/drigoni/repository/volta/ \
    -u root\
    --runtime=nvidia \
    -it \
    $VERSION \
    '/bin/bash'
else
  echo "To be implemented."
fi