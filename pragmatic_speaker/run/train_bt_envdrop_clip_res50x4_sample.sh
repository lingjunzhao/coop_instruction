#!/bin/bash 
 
set -x                                                                                              
                                                                                                                                                                                                      
name=bt_clip_listener_res50x4_rs615

flag="--attn soft --train auglistener --selfTrain
      --aug tasks/R2R/data/aug_paths.json
      --speaker snap/speaker_clip_res50x4/state_dict/best_val_unseen_bleu
      --load snap/clip_listener_res50x4_rs615/state_dict/best_val_unseen
      --angleFeatSize 128
      --features img_features/CLIP-ResNet-50x4-views.tsv
      --feature_size 640
      --accumulateGrad
      --featdropout 0.4
      --train_sampling 0.9
      --seed 615
      --subout max --optim rms --lr 1e-4 --iters 200000 --maxAction 35"

mkdir -p snap/$name

python3 -u r2r_src/train.py $flag --name $name
