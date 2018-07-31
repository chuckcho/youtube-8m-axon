from collections import defaultdict, Counter
import glob
from itertools import izip
import os
import subprocess
import sys
import time

if os.environ.get('INFDIR'):
  inference_dir = os.environ['INFDIR']
else:
  #inference_dir = '/tmp/__MODEL_VAULT__'
  inference_dir = '/media/TB2/chuck/__MODEL_VAULT__'

significant_digits = 6
top_k = 50
show_progress_iter = 20000

submission_ready = False
if submission_ready:
  top_k = 20

def file_len(fname):
  p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
  result, err = p.communicate()
  if p.returncode != 0:
      raise IOError(err)
  return int(result.strip().split()[0])

def do_stuff(model_pred, output_file, total_weight=None):
  total_num_vids = file_len(model_pred[0])
  if total_weight is None:
    total_weight = len(model_pred)
  files = [open(filename) for filename in model_pred]
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

    for l in lines:
      id, r = l.split(',')
      r = r.split(' ')
      assert len(r) % 2 == 0, "Number of fields is odd. That's odd!"
      for i in range(0, len(r), 2):
        k = int(r[i])
        v = int(10**(significant_digits - 1) * float(r[i+1]))
        blend[k] += v
    l = ' '.join(['{} {:{}f}'.format(t[0]
            , float(t[1]) / 10 ** (significant_digits - 1) / total_weight
            , significant_digits) for t in blend.most_common(top_k)])
    out_fp.write(','.join([id, l + '\n']))

model_pred = glob.glob(os.path.join(inference_dir, "*-sorted-noheader.csv"))
print "[Info] model_pred={}".format(model_pred)
do_stuff(model_pred, 'inference-ens{}.csv'.format(len(model_pred)))
