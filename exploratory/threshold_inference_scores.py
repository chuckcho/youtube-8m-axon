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
show_label_names = True
inf_csv = 'val_gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-2-cp64140.csv'
label_csv = 'val_gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-2-cp64140-threshold_0.5-label_names.txt'

if show_label_names:
    import pandas as pd
    labels_df = pd.read_csv('../label_names_2018_fixed.csv')

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
    valid_labels = sorted(valid_labels)

    if show_label_names:
        label_names = []
        for label_id in valid_labels:
            # some labels ids are missing so have put try/except
            try:
                label_names.append(str(labels_df[labels_df['label_id']==label_id]['label_name'].values[0]))
            except:
                continue
        out_fid.write(vid + ": " + str(label_names) + "\n")
    else:
        out_fid.write(vid + ": " + " ".join(str(x) for x in valid_labels) + "\n")
