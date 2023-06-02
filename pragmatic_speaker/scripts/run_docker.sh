#!/bin/bash

DATA_DIR=/Users/cathvoilet/Documents/CS_umd/Navigation/Matterport3DGymWithSpeaker/code/data/
CODE_DIR=`pwd`

echo "Data dir: $DATA_DIR"
echo "Code dir: $CODE_DIR"

docker run -it \
	       --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/R2R/code/data/v1/scans \
               --mount type=bind,source=$DATA_DIR,target=/root/mount/R2R/data \
               --volume $CODE_DIR:/root/mount/R2R/code \
               kxnguyen/pytorch:pytorch17-transformers 
