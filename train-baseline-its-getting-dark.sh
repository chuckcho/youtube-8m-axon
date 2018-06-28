#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
python train.py \
  --frame_features \
  --model=LstmModel \
  --feature_names='rgb,audio' \
  --feature_sizes='1024,128' \
  --train_data_pattern=/media/6TB/videos/yt8m/frame/train???[13579].tfrecord \
  --train_dir ./baseline-lstm \
  --batch_size=256 \
  --start_new_model
