#!/usr/bin/env python

'''
Some basic analysis / experiments based on two most common labels ("Game" and
"Video game").
The purpose of this analysis/experiment was to verify if a simple heuristics can
work beneficially. If that's the case, we can pursue label relationships with
more sophisticated/principled method/algorithm.
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from itertools import izip
import operator
import cPickle as pickle

def pairwise(t):
  it = iter(t)
  return izip(it,it)

inf_i = "/home/chuck/tmp/Youtube-8M-WILLOW/test-gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-pduan-cp_274278.csv"

'''
max_l = 400
smoothing = 1.0
inf_o = "/home/chuck/tmp/Youtube-8M-WILLOW/baseline-pduan-cp_274278-rj5.csv"

max_l = 5
smoothing = 3.0
inf_o = "/home/chuck/tmp/Youtube-8M-WILLOW/baseline-pduan-cp_274278-rj6.csv"

max_l = 100
smoothing = 0.5
inf_o = "/home/chuck/tmp/Youtube-8M-WILLOW/baseline-pduan-cp_274278-rj7.csv"
'''

max_l = 50
max_best_l = 5
threshold = 0.5
high_score_scale = 2.0
low_score_scale = 1.0
inf_o = "/home/chuck/tmp/Youtube-8M-WILLOW/baseline-pduan-cp_274278-rj8.csv"

max_l = 500
max_best_l = 300
threshold = 0.5
high_score_scale = 1.03
low_score_scale = 0.97
inf_o = "/home/chuck/tmp/Youtube-8M-WILLOW/baseline-pduan-cp_274278-rj9.csv"

new_thresh_file = 'new_threshold.p'
new_thresh = pickle.load(open(new_thresh_file, "r"))

good_labels = sorted(new_thresh.keys(), key=lambda i: new_thresh[i][1], reverse=True)[:max_best_l]

'''
out_fid = open(inf_o,'w')
with open(inf_i, 'r') as fid:
  header = fid.next()
  out_fid.write(header)
  for line in fid:
    vid, label_scores = line.split(',')
    label_scores = pairwise(label_scores.split(' '))

    all_predictions = dict()
    for label_score in label_scores:
      l, s = label_score
      l = int(l)
      s = float(s)
      #print "label={}, score={}".format(l, s)
      if l < max_l + 1:
        all_predictions[l] = s * (0.5/new_thresh[l][0]/smoothing)
      else:
        all_predictions[l] = s

    label_score_pairs_string = ""
    for l, s in sorted(all_predictions.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        label_score_pairs_string += "{} {:.6f} ".format(l, s)

    out_fid.write(vid + "," + label_score_pairs_string[:-1] + "\n")
'''

out_fid = open(inf_o,'w')
with open(inf_i, 'r') as fid:
  header = fid.next()
  out_fid.write(header)
  for line in fid:
    vid, label_scores = line.split(',')
    label_scores = pairwise(label_scores.split(' '))

    all_predictions = dict()
    for label_score in label_scores:
      l, s = label_score
      l = int(l)
      s = float(s)
      #print "label={}, score={}".format(l, s)
      if l in good_labels:
        if s >= threshold:
          all_predictions[l] = min(s * high_score_scale, 1.0)
        else:
          all_predictions[l] = s * low_score_scale
      else:
        all_predictions[l] = s

    label_score_pairs_string = ""
    for l, s in sorted(all_predictions.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        label_score_pairs_string += "{} {:.6f} ".format(l, s)

    out_fid.write(vid + "," + label_score_pairs_string[:-1] + "\n")
