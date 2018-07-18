#!/usr/bin/env python

'''
Read all labeled data (tfrecord's) and save a list of "video id, labels" as a
CSV file. train_labels.csv, and validate_labels.csv downloaded from google [1][2]
don't seem to contain all examples.

[1] http://us.data.yt8m.org/2/ground_truth_labels/train_labels.csv
[2] http://us.data.yt8m.org/2/ground_truth_labels/validate_labels.csv
'''

import glob
import tensorflow as tf

label_file = "./all_labels_new.csv"
records = glob.glob("/media/6TB/video/yt8m-v2/video/*a*.tfrecord")

# we'll only consider a single label for a given video
with open(label_file, 'w') as fid:
    for record in records:
        for example in tf.python_io.tf_record_iterator(record):
            tf_example = tf.train.Example.FromString(example)
            vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
            labels = tf_example.features.feature['labels'].int64_list.value
            label_str = ' '.join([str(x) for x in labels])
            fid.write('{},{}\n'.format(vid_id, label_str))
