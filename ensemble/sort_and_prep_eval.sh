#!/usr/bin/env bash

inference_dir=/media/TB2/chuck/__MODEL_VAULT__
post_fix=sorted-noheader

# exclude the header (first line) and sort accordning to video id
unprocessed_inference=$(ls -1 ${inference_dir}/*.csv | egrep -v ${post_fix})
for i in ${unprocessed_inference}; do
  echo Prepping $i...
  filename="${i##*/}"
  extension="${filename##*.}"
  filename="${filename%.*}"
  o=${inference_dir}/${filename}-${post_fix}.csv

  if [[ -f $o ]]; then
    echo Output file=$o already exist. Skipping...
    continue
  fi

  cat $i | tail -n+2 | sort -k1,1 -t, > $o
done
