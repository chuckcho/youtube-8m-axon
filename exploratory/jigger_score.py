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

def pairwise(t):
  it = iter(t)
  return izip(it,it)

inf_i = "/home/chuck/tmp/Youtube-8M-WILLOW/test-gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-pduan-cp_274278.csv"

# logic: if a video has class 'Video game' > .5,
vg_thresh = 0.5
all_game_scores = []

game_thresh = 0.8
low_game_scale = 0.25
high_game_scale = 1.0
inf_o = "/home/chuck/tmp/Youtube-8M-WILLOW/baseline-pduan-cp_274278-rj.csv"

game_thresh = 0.85
low_game_scale = 0.0
high_game_scale = 100.0
inf_o = "/home/chuck/tmp/Youtube-8M-WILLOW/baseline-pduan-cp_274278-rj2.csv"

game_thresh = 0.5
low_game_scale = 0.9
high_game_scale = 1.1
inf_o = "/home/chuck/tmp/Youtube-8M-WILLOW/baseline-pduan-cp_274278-rj3.csv"

game_thresh = 0.85
low_game_scale = 0.8
high_game_scale = 1.2
inf_o = "/home/chuck/tmp/Youtube-8M-WILLOW/baseline-pduan-cp_274278-rj4.csv"


# plot distribution of "Game"-score
if False:
  with open(inf_i, 'r') as fid:
    next(fid)
    for line in fid:
      vid, label_scores = line.split(',')
      label_scores = pairwise(label_scores.split(' '))

      all_predictions = dict()
      for label_score in label_scores:
        l, s = label_score
        #print "label={}, score={}".format(l, s)
        all_predictions[int(l)] = float(s)

      if 1 in all_predictions and all_predictions[1] > vg_thresh:
        if 0 in all_predictions:
          all_game_scores.append(all_predictions[0])
        else:
          all_game_scores.append(0.0)

  print "len(all_game_scores)={}".format(len(all_game_scores))
  print "min(all_game_scores)={}".format(min(all_game_scores))
  print "max(all_game_scores)={}".format(max(all_game_scores))
  print "median(all_game_scores)={}".format(np.median(all_game_scores))

  histogram_fig = 'game_class_score_histogram.png'
  binwidth = 0.01

  fig = pl.hist(all_game_scores, density=False, bins=np.arange(0.0, 1.0 + binwidth, binwidth))
  pl.title('Histogram of "Game" scores for videos with "Video game">0.5')
  pl.xlabel('"Game" score')
  pl.ylabel('Frequency')
  pl.savefig(histogram_fig)

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
      #print "label={}, score={}".format(l, s)
      all_predictions[int(l)] = float(s)

    rejiggered_all_predictions = all_predictions
    if 1 in all_predictions and all_predictions[1] > vg_thresh:
      if 0 in all_predictions:
        if all_predictions[0] >= game_thresh:
          rejiggered_all_predictions[0] = min(1.0, all_predictions[0] * high_game_scale)
        else:
          rejiggered_all_predictions[0] = all_predictions[0] * low_game_scale

    label_score_pairs_string = ""
    for l, s in sorted(rejiggered_all_predictions.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        label_score_pairs_string += "{} {:.6f} ".format(l, s)

    out_fid.write(vid + "," + label_score_pairs_string[:-1] + "\n")
