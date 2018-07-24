#!/usr/bin/env bash

# data_path should be good for ALL seattle GPU boxes
data_path="/media/6TB/video/yt8m-v2/frame"

# use ALL test examples
#axon_test_set="${data_path}/test????.tfrecord"

# or, try this for sanity check before running fully
#axon_test_set="${data_path}/test000?.tfrecord"

# or, in otder to train a model for ensembling (optimum model weights)
# we need inference on validate???5 data which we have labels for
axon_test_set="${data_path}/validate???5.tfrecord"
output_prefix=inference-on-validate5

# be courteous, don't claim all GPU's! ;)
export CUDA_VISIBLE_DEVICES=1

# less verbose?
export TF_CPP_MIN_LOG_LEVEL=3

# inference.py assumes eval.py has been already run, and created
# inference_model.* files. If these files are not there, we'll symlinks from
# check_point
train_basedir=/media/6TB/video/yt8m-v2/__MODEL_VAULT__
top_k=50

for train_dir in ${train_basedir}/*/
do
  echo -------------------------------------------------------------------------
  train_dir=${train_dir%/}

  # check if checkpoint file is there
  if [ ! -f ${train_dir}/checkpoint ]; then
    echo checkpoint file inside ${train_dir} does not exist. skipping...
    continue
  fi

  # check if optino_arg file is there
  if [ ! -f ${train_dir}/option_arg.txt ]; then
    echo option argument file inside ${train_dir} does not exist. skipping...
    continue
  fi

  # grab check_point iteration from checkpoint file
  check_point=$(cat ${train_dir}/checkpoint| egrep "^model_checkpoint_path" | head -n 1 | sed -e 's/.*ckpt\-//' -e 's/"//g')

  # model_name is just basename of current train_dir
  model_name=${train_dir##*/}

  # consistent inference csv filename
  output_file=${train_basedir}/${output_prefix}-${model_name}-cp${check_point}-top${top_k}.csv

  # grab additional model-specific input arguments
  option_args=$(cat ${train_dir}/option_arg.txt)

  # if inference csv is there, don't re-run
  if [ -f ${output_file} ]; then
    echo inference output=${output_file} already exists. skipping...
    continue
  fi

  # back up checkpoint file as it will be overwritten
  cp -a ${train_dir}/checkpoint ${train_dir}/checkpoint.bak

  # run eval.py just to get inference_model.* and strip all the hard-coded
  # tfrecord file names
  # TODO: this somehow fails on one of the GRU models. :( why why?
  #       if inference_model.* files are symlinked (instead of created as
  #       by-products of eval.py), inference.py runs fine.
  echo ==============
  echo Running
  echo python eval.py \
    --eval_data_pattern=${data_path}/validate0005.tfrecord \
    --train_dir=${train_dir} \
    ${option_args} \
    --batch_size=128
  echo ==============

  python eval.py \
    --eval_data_pattern=${data_path}/validate0005.tfrecord \
    --train_dir=${train_dir} \
    ${option_args} \
    --batch_size=128

  # check if eval was run correctly
  rc=$?
  if [[ $rc != 0 ]]; then
    echo eval.py did not run correctly. exit code=${rc}. exitting...
    continue
  fi

  # revert the checkpoint file
  cp -a ${train_dir}/checkpoint.bak ${train_dir}/checkpoint

  # now run inference
  # TODO: inference using GPU
  echo ==============
  echo Running
  echo python inference.py \
    --output_file=${output_file} \
    --input_data_pattern=${axon_test_set} \
    --train_dir=${train_dir} \
    ${option_args} \
    --top_k=${top_k} \
    --batch_size=512
  echo ==============

  python inference.py \
    --output_file=${output_file} \
    --input_data_pattern=${axon_test_set} \
    --train_dir=${train_dir} \
    ${option_args} \
    --top_k=${top_k} \
    --batch_size=512

  # check if inference was run correctly
  rc=$?
  if [[ $rc != 0 ]]; then
    echo inference.py did not run correctly. exit code=${rc}. exitting...
    echo most likely the model was trained with https://github.com/antoine77340/Youtube-8M-WILLOW and not with https://git.taservs.net/axon-research/youtube-8m
    continue
  fi

  # delete junks
  rm -f ${train_dir}/events.out.tfevents.*

done
