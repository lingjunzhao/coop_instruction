#!/bin/bash 
 
set -x                                                                                              
                                                                                                                                                                                                      
name=t5_speaker_rs100
flag="--attn soft --angleFeatSize 128
      --train speaker
      --features img_features/CLIP-ViT-B-32-views.tsv
      --feature_size 512
      --rnnDim 512
      --speaker_model t5
      --transformer_dropout 0.3
      --batchSize 32
      --num_transformer_layers 2
      --seed 100
      --subout max --dropout 0.6 --optim adamW --lr 1e-4 --iters 160000 --maxAction 35"

mkdir -p snap/$name

python3 -u r2r_src/train.py $flag --name $name
