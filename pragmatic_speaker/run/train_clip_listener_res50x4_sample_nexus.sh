#!/bin/bash 
 
set -x                                                                                              
                                                                                                  
name=clip_listener_res50x4_rs615

flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --features img_features/CLIP-ResNet-50x4-views.tsv
      --feature_size 640
      --feedback sample
      --mlWeight 0.2
      --train_sampling 0.9
      --seed 615
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35"

mkdir -p snap/$name

python3 -u r2r_src/train.py $flag --name $name

