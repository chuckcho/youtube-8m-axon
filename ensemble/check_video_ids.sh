#!/usr/bin/env bash

# check if all video_ids match (after sorting)

# if env var is set, use it. otherwise specify here manually.
if [[ ! -z "${INFDIR}" ]]; then
  inference_dir=${INFDIR}
else
  inference_dir=/media/TB2/chuck/__MODEL_VAULT__
fi
post_fix=sorted-noheader

# exclude the header (first line) and sort accordning to video id
file_count=$(ls -1 ${inference_dir}/*.csv | egrep ${post_fix} | wc -l)
if [[ ${file_count} -le 1 ]]; then
  echo file_count=${file_count}. nothing to do
  exit -1
fi

processed_inference=$(ls -1 ${inference_dir}/*.csv | egrep ${post_fix})

# first pass: check number of lines
num_lines=""
for i in ${processed_inference}; do
  echo Checking $i...
  num_lines_this_file=$(wc -l < $i)
  if [[ -z ${num_lines} ]]; then
    num_lines=${num_lines_this_file}
    continue
  else
    if [[ ${num_lines} != ${num_lines_this_file} ]]; then
      echo Number of lines differ. This file has ${num_lines_this_file} lines, but previous one\(s\) had ${num_lines}. Panic\!
      exit -2
    fi
  fi
done

# second pass: check if video ID's match up
echo Number of lines all agree. Now checking if video ID\'s all match up.
out=/tmp/video-ids.txt
rm -f ${out}

for i in ${processed_inference}; do
  echo Checking $i...
  filename="${i##*/}"
  extension="${filename##*.}"
  filename="${filename%.*}"

  if [[ ! -f ${out} ]]; then
    cat ${i} | cut -d, -f1 > ${out}
    continue
  else
    cat ${i} | cut -d, -f1 | diff -- -  ${out} >& /dev/null
    error=$?
    if [[ $error != "0" ]]; then
      echo vid_id\'s didn\'t match. Panic\!
      exit -3
    fi
  fi
done

