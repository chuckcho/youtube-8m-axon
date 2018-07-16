import tensorflow as tf
import numpy as np
import glob
from itertools import izip
import cPickle as pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

'''
A quick and dirty tool to get GAP result for csv-formatted inference output.
This may be useful if you blend CSV results from multiple models into a single
CSV, and wish to evaluate that results in terms of GAP. You'll of course need
ground-truth labels, hence for train and validate data sets.
'''

def max_a_posteriori(p_samples, n_samples):
  # brute-force calculation of MAP threshold
  lowest_Pr_err = 1e9 # should be larger than number of samples (videos)
  for thresh in np.arange(0.4, 0.6, 0.005):
    Pr_err = sum(x<thresh for x in p_samples) + sum(x>thresh for x in n_samples)
    if Pr_err < lowest_Pr_err:
      lowest_Pr_err = Pr_err
      best_threshold = thresh

  return best_threshold, lowest_Pr_err

def pairwise(t):
  it = iter(t)
  return izip(it,it)

records = glob.glob("/media/6TB/video/yt8m-v2/video/validate???5.tfrecord")
csv = "val_gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-ens2.csv"

num_classes = 3862
batch_size = 1
top_k = 20
num_videos = 111022

score_dist_file = "all_score_distribution.p"
if os.path.isfile(score_dist_file):
  distrib = pickle.load(open(score_dist_file, "r"))
else:
  all_labels = dict()
  for record in records:
    for example in tf.python_io.tf_record_iterator(record):
      tf_example = tf.train.Example.FromString(example)
      vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
      labels = tf_example.features.feature['labels'].int64_list.value
      all_labels[vid_id] = labels

  # distrib(label) is a tuple of lists, ( [pos_scores], [neg_scores] )
  distrib = dict()
  with open(csv, 'r') as fid:
    next(fid)
    for line in fid:
      vid, label_scores = line.split(',')
      label_scores = pairwise(label_scores.split(' '))
      assert vid in all_labels, "records and csv don't match"

      labels_top_k = set()
      for label_score in label_scores:
          l, s = label_score
          l = int(l)
          s = float(s)
          if l not in distrib:
            distrib[l] = ( [], [] )
          if l in all_labels[vid]:
            distrib[l][0].append(s)
          else:
            distrib[l][1].append(s)
          labels_top_k.add(l)
      #remaining_labels = set(range(num_classes)).difference(labels_top_k)
      #for l in remaining_labels:
      #  if l not in distrib:
      #    distrib[l] = ( [], [] )
      #  distrib[l][1].append(0.0)

    #print "distrib={}".format(distrib)

  pickle.dump(distrib, open(score_dist_file, "wb"))

labels_df = pd.read_csv(os.path.join('..', 'label_names_2018_fixed.csv'))
include_zero_scores = True

# do some plotting!
new_thresh = dict()
for l in distrib:
  # some labels ids are missing so have put try/except
  try:
    label_name = str(labels_df[labels_df['label_id']==l]['label_name'].values[0]).decode('ascii', 'ignore')
  except:
    label_name = '[unavailable]'
  print "Plotting histograms for label_id={}, label_name=\"{}\"".format(l, label_name)
  plot_filename = 'label_{:04d}_histogram.png'.format(l)
  p_samples = distrib[l][0]
  n_samples = distrib[l][1]

  if include_zero_scores:
    num_samples = len(p_samples) + len(n_samples)
    num_remaining_samples = num_videos - num_samples
    n_samples += [0.0] * num_remaining_samples

  optimum_thresh, Pr_err = max_a_posteriori(p_samples, n_samples)
  Pr_err = Pr_err / float(len(p_samples) + len(n_samples)) * 100
  new_thresh[l] = (optimum_thresh, Pr_err)
  print "[Debug] best threshold={:.3f}, Pr_err={:.2f}".format(optimum_thresh, Pr_err)
  bins = np.linspace(-0.01, 1.01, 50)

  plt.hist(p_samples, bins, alpha=0.5, label='+ (#vid={})'.format(len(p_samples)))
  plt.hist(n_samples, bins, alpha=0.5, label='-')
  plt.yscale('log', nonposy='clip')
  plt.legend(loc='upper right')
  if len(p_samples) > 0:
    pct_p_samples_higher_than_thresh = 100.0 * sum(p_sample >= 0.5 for p_sample in p_samples)/len(p_samples)
  else:
    pct_p_samples_higher_than_thresh = -1.0
  plt.title('label="{}" score dist ({:.1f}% +score>=0.5, best thresh={:.3f})'.format(label_name, pct_p_samples_higher_than_thresh, optimum_thresh))
  plt.savefig(plot_filename)
  plt.close()

new_thresh_file = 'new_threshold.p'
pickle.dump(new_thresh, open(new_thresh_file, "wb"))
