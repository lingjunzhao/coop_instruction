#!/bin/bash --login

set -x

name=t5_speaker

flag="--attn soft --angleFeatSize 128
      --train validspeakerdecode
      --features img_features/CLIP-ViT-B-32-views.tsv
      --input_decode_json_file datasets/val_seen.json 
      --feature_size 512
      --speaker_model t5
      --decode_top_p 0.95
      --subout max --dropout 0.6 --optim adam --lr 1e-4 --iters 80000 --maxAction 35"


python3 -u r2r_src/train.py $flag --name $name
