This directory
--------------
This directory contains some useful scripts to do average ensembling on a set of big inference files.

Steps to generate a final "ensembled" inference
------------------------------------------------
1. Copy all inference csv files into a single directory
2. Prep csv's by editting (`inference_dir`) and running `sort_and_prep_eval.sh`
3. Run sanity check (if video ID's match up) `check_video_ids.sh`
4. Run a python script `file_averaging.py`
