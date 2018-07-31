This directory contains some scripts to do average ensembling on a set of big inference files.

Steps to generated a final "ensembled" inference
================================================
1. Copy all inference csv files into a single directory
2. Prep csv's by editting (`inference_dir`) and running `sort_and_prep_eval.sh`
3. Run Sanity check (if video ID's macch up) `check_video_ids.sh`
4. Run a python script `mmap_file_averaging.py`
