#!/usr/bin/env bash

# data_path should be good for ALL seattle GPU boxes
data_path="/media/6TB/video/yt8m-v2/frame"

# training set for experimental/exploratory stage
# see: https://axon.quip.com/bOpyAw3mGmb3/YouTube-8M-Axon-Official-TrainValidate-Set
axon_train_set="${data_path}/train???[13579].tfrecord"

# be courteous, don't claim all GPU's! ;)
export CUDA_VISIBLE_DEVICES=0

python willow/train.py \
  --num_gpu=1 \
  --train_data_pattern="${axon_train_set}" \
  --train_dir=willow/gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-regularize \
  --model=NetVLADModelLF \

