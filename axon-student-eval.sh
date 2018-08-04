#!/usr/bin/env bash

# data_path should be good for ALL seattle GPU boxes
data_path="/media/6TB/video/yt8m-v2/frame"

# validation set for experimental/exploratory stage
# see: https://axon.quip.com/bOpyAw3mGmb3/YouTube-8M-Axon-Official-TrainValidate-Set
axon_val_set="${data_path}/validate0005.tfrecord"

# be courteous, don't claim all GPU's! ;)
export CUDA_VISIBLE_DEVICES=1

train_dir=distill-vlad-40-1p0-ssr-ensemble
check_point=647000

python eval.py \
  --eval_data_pattern=${axon_val_set} \
  --train_dir=${train_dir} \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=800 \
  --netvlad_relu=False \
  --iterations=300 \
  --gating=True \
  --moe_l2=1e-6 \
  --moe_prob_gating=True \
  --run_once=True \
  --batch_size=100 \
  --check_point=${check_point}
