#!/bin/bash

set -x

vote_name="10resbt"  

# change dataset name and file if you want to use some other candidate instructions
dataset_name="t5-sampled-val-seen"
dataset_file="speaker_t5_clip_sampled_nucleus_val_seen.json"

exp_dir="experiments/pi_vote-sample-${vote_name}_dataset-${dataset_name}/"
mkdir -p $exp_dir

# change voting_agents if you want to use some other listeners
voting_agents=("bt_clip_listener_res50x4_rs500" "bt_clip_listener_res50x4_rs505" "bt_clip_listener_res50x4_rs510" "bt_clip_listener_res50x4_rs515" "bt_clip_listener_res50x4_rs600" "bt_clip_listener_res50x4_rs605" "bt_clip_listener_res50x4_rs610" "bt_clip_listener_res50x4_rs615" "bt_clip_listener_res50x4_rs620" "bt_clip_listener_res50x4_rs625")

echo "Voting agents:"
echo ${voting_agents[*]}

for agent in "${voting_agents[@]}"; do
    flag="--attn soft --train eval_listener_outputs
      --featdropout 0.3
      --angleFeatSize 128
      --features img_features/CLIP-ResNet-50x4-views.tsv
      --feature_size 640
      --submit
      --load snap/${agent}/state_dict/best_val_unseen
      --speaker_output_files speaker_outputs/${dataset_file}
      --feedback sample
      --decode_feedback sample
      --mlWeight 0.2
      --seed 1
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35"
    name="${agent}_pi_vote-sample_dataset-${dataset_name}"

    mkdir -p snap/$name
    python3 -u r2r_src/train.py $flag --name $name
done


python3 prag_inf/vote_instructions.py --output_exp "$exp_dir" --input_path "pi_vote-sample_dataset-${dataset_name}/${dataset_file}" --metric "avg" --no_prob 1 --result_sample 10 -input_exps "${voting_agents[@]}"
