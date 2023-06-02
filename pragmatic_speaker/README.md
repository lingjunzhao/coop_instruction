The following steps install Matterport Environment for grounded instruction generation task, as well as equipping the instruction generation model with theory-of-mind capability. It is based on the codebase of [How Much Can CLIP Benefit Vision-and-Language Tasks?](https://arxiv.org/abs/2107.06383).


## Install environment

Clone this repo. Follow the steps [here](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-VLN) to install the simulator environments. Follow the steps to download Pre-Computed Image Features (CLIP ViT and CLIP-Res50x4), and put under `img_features/`.



## Decode candidate instructions

1. Download the pretrained speaker model (EncDec-Transformer Base Speaker) [here](https://drive.google.com/file/d/1gDL0yIgJsDSC7Y-iIgC9zmgLpKaoan7Z/view?usp=share_link). Unzip and put the directory under `snap/`


2. Run the following script to decode greedy instruction and 10 random instructions:
```
run/decode_speaker_t5_vit.sh
```

The generated instructions are stored as `generated_instr` in the output json files.

3. Alternatively, you could skip step 1&2, and use the decoded instructions:
`speaker_outputs/speaker_t5_clip_sampled_nucleus_val_seen.json`



## Equip speaker with theory-of-mind capability

1. Download ensemble of 10 EnvDrop-CLIP listeners [here](https://drive.google.com/file/d/1TwXmeWi9FFkFXqfe7Pws1ZtyEjfU7FNZ/view?usp=share_link). Unzip and put the directories under `snap/`.
E.g. `snap/bt_clip_listener_res50x4_rs500/`

2. Run the following script to use ensemble listeners as ToM model:
```
run/pi_voting_sample.sh
```


## Train base speaker from scratch

If you want to train a EncDec-Transformer Base Speaker from scratch (not neccesary), run the following script:
```
train_speaker_t5_vit.sh

```
