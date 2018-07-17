#!/usr/bin/env bash

# data_path should be good for ALL seattle GPU boxes
data_path="/media/6TB/video/yt8m-v2/frame"

# training set for experimental/exploratory stage
# see: https://axon.quip.com/bOpyAw3mGmb3/YouTube-8M-Axon-Official-TrainValidate-Set
axon_train_set="${data_path}/train????.tfrecord,${data_path}/validate???[012346789].tfrecord"

# be courteous, don't claim all GPU's! ;)
export CUDA_VISIBLE_DEVICES=0

python train.py \
  --num_gpu=1 \
  --train_data_pattern="${axon_train_set}" \
  --train_dir=willow1-gatednetvladLF-full-minus-val5 \
  --model=NetVLADModelLF \
  --frame_features=True \
  --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --netvlad_relu=False \
  --iterations=300 \
  --gating=True \
  --moe_l2=1e-6 \
  --moe_prob_gating=True \
  --base_learning_rate=0.0002 \
  --learning_rate_decay=0.8 \
  --max_step=350000 \
  --batch_size=64
