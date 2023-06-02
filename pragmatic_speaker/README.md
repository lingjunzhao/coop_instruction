The following steps install Matterport Environment for grounded instruction generation task, as well as equipping the instruction generation model with theory-of-mind capability.


## Install environment

Clone this repo. Follow the steps "[here](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-VLN)" to install the simulator environments. Follow the steps to download Pre-Computed Image Features (CLIP ViT and CLIP-Res50x4), and put under `img_features/`



## Decode candidate instructions

1. Download pretrained speaker model (Base Encoder-decoder Transformer Speaker model). 

Unzip and put the directory under `snap/`


2. Run script to decode greedy instruction and 10 random instructions:
run/decode_speaker_t5_vit.sh

The generated instructions are stored as "generated_instr" in the output json files.

3. Alternatively, skip downloading pretrained speaker model, and use the decoded instructions:
speaker_outputs/speaker_t5_clip_sampled_nucleus_val_seen.json



## Equip speaker with theory-of-mind capability

1. Download ensemble of 10 listeners. Unzip and put the directories under `snap/`.
e.g. `snap/bt_clip_listener_res50x4_rs500/`

2. Run script to use ensemble listeners as ToM model:
run/pi_voting_sample.sh

