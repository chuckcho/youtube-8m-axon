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
# data_path for batch ai cluster
#data_path="/mnt/batch/tasks/shared/LS_root/mounts/youtube8m/frame"

# use ALL test examples
axon_test_set="${data_path}/test????.tfrecord"

# be courteous, don't claim all GPU's! ;)
export CUDA_VISIBLE_DEVICES=3

# inference.py assumes eval.py has been already run, and created
# inference_model.* files. If these files are not there, we'll symlinks from
# check_point
root_dir=.
train_dir=distill-vlad-40-1p0-ssr-ensemble
#check_point=370010
check_point=647000
top_k=20

if [ ! -f ${root_dir}/${train_dir}/inference_model.meta ]; then
  cd ${root_dir}/${train_dir}
  ln -s model.ckpt-${check_point}.meta                inference_model.meta
  ln -s model.ckpt-${check_point}.index               inference_model.index
  ln -s model.ckpt-${check_point}.data-00000-of-00001 inference_model.data-00000-of-00001
  cd -
fi

python inference.py \
  --output_file=${root_dir}/test-${train_dir}-cp${check_point}-top${top_k}.csv \
  --input_data_pattern=${axon_test_set} \
  --train_dir=${root_dir}/${train_dir} \
  --top_k=${top_k} \
  --batch_size=1024 \
  --output_model_tgz=${root_dir}/${train_dir}/${train_dir}-cp${check_point}.tgz \
