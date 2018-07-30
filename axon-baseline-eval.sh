#!/usr/bin/env bash

# data_path should be good for ALL seattle GPU boxes
data_path="/media/6TB/video/yt8m-v2/frame"

# validation set for experimental/exploratory stage
# see: https://axon.quip.com/bOpyAw3mGmb3/YouTube-8M-Axon-Official-TrainValidate-Set
axon_val_set="${data_path}/validate???5.tfrecord"

# be courteous, don't claim all GPU's! ;)
export CUDA_VISIBLE_DEVICES=0,1,2,3

python eval.py \
  --eval_data_pattern=${axon_val_set} \
  --train_dir=distill-vlad-32-1p0 \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=800 \
  --moe_l2=1e-6 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --run_once=True \
  --batch_size=80 \
  --check_point=170010
  #--input_model_tgz=tgz_model
  #--lightvlad=True \
