#!/usr/bin/env python
from itertools import izip

'''
Given an inference CSV file, apply score threshold and save a text file with a
list of
    video_id: [label1 label2 leval3...]
where score of labelx exceeds the given threshold
'''

def pairwise(t):
  it = iter(t)
  return izip(it,it)

threshold = .5
inf_csv = 'val_gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-ens2.csv'
label_csv = 'val_gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-ens2-label-threshold_0.5.txt'

out_fid = open(label_csv, 'w')
with open(inf_csv, 'r') as fid:
  header = fid.next()
  #out_fid.write(header)
  for line in fid:
    vid, label_scores = line.split(',')
    label_scores = pairwise(label_scores.split(' '))

    valid_labels = []
    for label_score in label_scores:
        l, s = label_score
        l = int(l)
        s = float(s)
        if s >= threshold:
            valid_labels.append(l)
    out_fid.write(vid + ": " + " ".join(str(x) for x  in sorted(valid_labels)) + "\n")
