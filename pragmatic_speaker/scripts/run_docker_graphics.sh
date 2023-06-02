#!/bin/bash

DATA_DIR=/Users/cathvoilet/Documents/CS_umd/Navigation/Matterport3DGymWithSpeaker/code/data/
CODE_DIR=`pwd`
MATTERPORT_DATA_DIR=/Users/cathvoilet/Documents/CS_umd/Navigation/data/v1/scans/

echo "Matterport3D data: $MATTERPORT_DATA_DIR"
echo "Data dir: $DATA_DIR"                                                         
echo "Code dir: $CODE_DIR"

xhost +  
docker run -it  \
               -e DISPLAY=host.docker.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix \
               --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/R2R/code/data/v1/scans,readonly \
               --mount type=bind,source=$DATA_DIR,target=/root/mount/R2R/data \
               --volume $CODE_DIR:/root/mount/R2R/code \
               kxnguyen/pytorch:pytorch17-transformers
