#!/usr/bin/env bash

# data_path should be good for ALL seattle GPU boxes
data_path="/media/6TB/video/yt8m-v2/frame"

# training set for experimental/exploratory stage
axon_train_set="${data_path}/train???[13579].tfrecord"

MODEL='NetVLADModelLF'
output_dir="teacher_model_predictions/$MODEL"
teacher_model_dir="/media/6TB/shwan/workspace/projects/youtube-8m/teacher_model"
#teacher_model_dir="gatednetvladLF-256k-1024-2-0002-300iter-norelu-basic-gatedmoe"

GPU_ID=3
#export CUDA_VISIBLE_DEVICES=$GPU_ID

if [ ! -d $output_dir ]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" python inference-pre-ensemble.py \
    --output_dir="${output_dir}" \
    --train_dir="${teacher_model_dir}" \
    --input_data_pattern="${axon_train_set}" \
    --frame_features=True \
    --feature_names="rgb,audio" \
    --feature_sizes="1024,128" \
    --model=$MODEL \
    --batch_size=2 \
    --file_size=4100
fi



    #--moe_num_mixtures=4 \
    #--deep_chain_layers=4 \
    #--deep_chain_relu_cells=128 \
 
