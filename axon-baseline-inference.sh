#!/usr/bin/env bash

##############################################################################
# if you submit the resulting inference CSV file to kaggle server, please document
# meticulously how you trained this model or how you ensembled multiple models.
# otherwise, we will easily lose track of all the good improvements we made, and
# won't be able to reproduce the results! VERY VERY IMPORTANT
# add an entry to:
# https://axon.quip.com/nzbHAlae4bK7/Youtube-8M-Submission-to-Leaderboard
##############################################################################

# data_path should be good for ALL seattle GPU boxes
data_path="/media/6TB/video/yt8m-v2/frame"

# use ALL test examples
axon_val_set="${data_path}/validate???5.tfrecord"

# be courteous, don't claim all GPU's! ;)
export CUDA_VISIBLE_DEVICES=3

train_dir="distill-vlad-16-0p95"

python inference.py \
  --output_file="${train_dir}.csv" \
  --input_data_pattern=${axon_val_set} \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --run_once=True \
  --top_k=50 \
  --batch_size=16 \
  --check_point=169066 \
  --output_model_tgz=True \
  --train_dir=${train_dir}
