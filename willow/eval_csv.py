import eval_util
import tensorflow as tf
import numpy as np
import glob
from itertools import izip

'''
A quick and dirty tool to get GAP result for csv-formatted inference output.
This may be useful if you blend CSV results from multiple models into a single
CSV, and wish to evaluate that results in terms of GAP. You'll of course need
ground-truth labels, hence for train and validate data sets.
'''

def pairwise(t):
  it = iter(t)
  return izip(it,it)

records = glob.glob("/media/6TB/video/yt8m-v2/video/validate???5.tfrecord")
csv = "val_gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-ens2.csv"

num_classes = 3862
batch_size = 1
top_k = 20
evl_metrics = eval_util.EvaluationMetrics(num_classes, top_k)

all_labels = dict()
for record in records:
  for example in tf.python_io.tf_record_iterator(record):
    tf_example = tf.train.Example.FromString(example)
    vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
    labels = tf_example.features.feature['labels'].int64_list.value
    all_labels[vid_id] = labels

evl_metrics.clear()
with open(csv, 'r') as fid:
  next(fid)
  for line in fid:
    vid, label_scores = line.split(',')
    label_scores = pairwise(label_scores.split(' '))
    assert vid in all_labels, "records and csv don't match"
    predictions = np.zeros((batch_size, num_classes))
    for label_score in label_scores:
        l, s = label_score
        predictions[0,int(l)] = float(s)
    labels = np.zeros((batch_size, num_classes))
    labels[0,all_labels[vid]] = 1

    # ignore loss (0.0)
    iteration_info_dict = evl_metrics.accumulate(predictions, labels, 0.0)
epoch_info_dict = evl_metrics.get()

print "GAP={}".format(epoch_info_dict['gap'])
