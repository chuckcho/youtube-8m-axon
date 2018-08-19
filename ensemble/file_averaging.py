from collections import defaultdict, Counter
import glob
from itertools import izip
import os
import subprocess
import sys
import time

model_weights = [
        0.2367,
        0.1508,
        0.1590,
        0.1000,
        0.1968,
        0.1306,
        0.0621,
        ]

scaler = float(len(model_weights)) / sum(model_weights)
model_weights = [ x * scaler for x in model_weights ]

if os.environ.get('INFDIR'):
  inference_dir = os.environ['INFDIR']
else:
  #inference_dir = '/tmp/__MODEL_VAULT__'
  inference_dir = '/media/TB2/chuck/__MODEL_VAULT__'
  inference_dir = '/media/6TB/video/yt8m-v2/__MODEL_VAULT__/inference-on-test'

significant_digits = 6
top_k = 50
show_progress_iter = 20000
use_model_weights = True

submission_ready = True
if submission_ready:
  top_k = 20

def file_len(fname):
  p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
  result, err = p.communicate()
  if p.returncode != 0:
      raise IOError(err)
  return int(result.strip().split()[0])

def get_model_ids(model_files):
  # get zero-based model_ids from a list of filenames
  # filenames are in either of these formats:
  # (1) test-model01-01-cp274278-top50.csv or
  # (2) willow1-2.csv
  model_ids = []
  for model_file in model_files:
    if "willow" in model_file:
      idx = model_file.index("willow") + len("willow")
    elif "model0" in model_file:
      idx = model_file.index("model0") + len("model0")
    else:
      assert False, "model file={} looks weird. panic!".format(model_file)
    model_id = int(model_file[idx]) - 1
    if model_id > 6 or model_id < 0:
      assert False, "model file={} has model_id={}. looks weird. panic!".format(model_file, model_id)
    model_ids.append(model_id)
  return model_ids

def do_stuff(model_pred, output_file, total_weight=None):
  total_num_vids = file_len(model_pred[0])
  if total_weight is None:
    total_weight = len(model_pred)
  files = [open(filename) for filename in model_pred]
  if use_model_weights:
    model_ids = get_model_ids(model_pred)
  out_fp = open(output_file, 'w')

  # add header
  if submission_ready:
    out_fp.write('VideoId,LabelConfidencePairs\n')

  start_time = time.time()
  for line_count, lines in enumerate(izip(*files)):
    blend = Counter()
    if line_count % show_progress_iter == 0:
      elapsed = (time.time() - start_time)
      throughput = float(line_count+1) / elapsed
      eta = elapsed * (float(total_num_vids) / (line_count+1) - 1) / 60.0
      print("[Info] Progress={:2.1f}% ({}/{}), Throughput={:.1f} lines/s, ETA={:.1f} min".format(
              float(line_count+1) / total_num_vids * 100,
              line_count+1,
              total_num_vids,
              throughput,
              eta,
              ))

    for model_idx, l in enumerate(lines):
      id, r = l.split(',')
      r = r.split(' ')
      assert len(r) % 2 == 0, "Number of fields is odd. That's odd!"
      for i in range(0, len(r), 2):
        k = int(r[i])
        v = int(10**(significant_digits - 1) * float(r[i+1]))
        if use_model_weights:
          w = model_weights[model_ids[model_idx]]
        else:
          w = 1.0
        blend[k] += v * w
    l = ' '.join(['{} {:{}f}'.format(t[0]
            , float(t[1]) / 10 ** (significant_digits - 1) / total_weight
            , significant_digits) for t in blend.most_common(top_k)])
    out_fp.write(','.join([id, l + '\n']))

model_pred = glob.glob(os.path.join(inference_dir, "*-sorted-noheader.csv"))
print "[Info] model_pred={}".format(model_pred)
do_stuff(model_pred, 'inference-ens{}.csv'.format(len(model_pred)))
