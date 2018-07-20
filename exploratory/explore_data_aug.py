#!/usr/bin/env python

'''
Read label data (a list of "video: [labels]") and find data augmentation ratio
per video
'''

from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

label_file = "./all_labels_new.csv"

vid_label = dict()
all_labels = []
with open(label_file, 'r') as f:
    for line in f:
        vid, labels = line.split(',')

        labels = labels.rstrip()
        if ' ' in labels:
            labels = [int(x) for x in labels.split(' ')]
        else:
            try:
                labels = [int(labels)]
            except:
                continue

        #print "vid={}, labels={}".format(vid, labels)
        vid_label[vid] = labels
        all_labels.extend(labels)

label_counter = Counter(all_labels)
counts = sorted(label_counter.values(), reverse=True)
labels_sorted_by_samples = [x[0] for x in label_counter.most_common()]
#print "label_count={}".format(counts)

print "#videos={}".format(len(vid_label))
plt.loglog(counts, label='original')
plt.title('# samples per class')
plt.grid(True)
plt.savefig('original_num_samples_per_class.png')

# based on ^ plot, ~100 classes have abundant samples (> 10K)
# so, we'll subsample for these classes, and will augment for all other
# classes

# multiplier for over-sampling regime
alpha = 1.0

new_all_labels = []

vid_count = 0
vid_label_sampling = dict()
oversampling_labels = set()

for vid in vid_label:
    labels = vid_label[vid]

    min_count = 1e7
    for label in labels:
        if label_counter[label] < min_count:
            min_count = label_counter[label]
            min_count_label = label

    min_count_label_index = labels_sorted_by_samples.index(min_count_label)

    # sub-sampling regime
    # 1e4 -> ratio = 1
    # ~4e5 -> ratio = 1/20
    if min_count > 1e4:
        ratio = int(min_count / (((np.log10(min_count) - 4.0) / 2.0 + 1) * 1e4))
        #print "vid={}, min_count={}, sub-sampling ratio={}, label={}, index={}, labels={}".format(vid, min_count, ratio, min_count_label, min_count_label_index, labels)

        # sub-sampled and selected
        if hash(vid) % ratio == 0:
            new_all_labels.extend(labels)
            vid_count += 1

    # over-sampling regime
    else:
        ratio = int((((np.log10(min_count) - 4.0) / 2.0 + 1) * 1e4) / min_count * alpha + 1)
        #print "vid={}, min_count={}, over-sampling ratio={}, label={}, index={}, labels={}".format(vid, min_count, ratio, min_count_label, min_count_label_index, labels)

        # over-sampled
        for _ in range(ratio):
            new_all_labels.extend(labels)
        vid_count += ratio
        oversampling_labels.add(min_count_label)

    vid_label_sampling[vid] = {
            'labels': labels,
            'min_count': min_count,
            'min_count_label': min_count_label,
            'min_count_label_index': min_count_label_index,
            'regime': 'subsampling' if min_count > 1e4 else 'oversampling',
            'ratio': ratio,
            'selected_if_subsampling': hash(vid) % ratio == 0 if min_count > 1e4 else None,
            }

'''
pickle.dump(vid_label_sampling, open('data_augmentation_info.p', 'w'))

new_label_counter = Counter(new_all_labels)
new_counts = sorted(new_label_counter.values(), reverse=True)
new_labels_sorted_by_samples = [x[0] for x in new_label_counter.most_common()]
#print "label_count={}".format(counts)

print "#original videos={}, after sub/over-sampling, #videos={}".format(len(vid_label), vid_count)
plt.loglog(new_counts, label='after sub/over-sampling')
plt.title('# samples per class after sub/over-sampling')
plt.grid(True)
plt.legend()
plt.savefig('new_num_samples_per_class.png')
'''

pickle.dump(oversampling_labels, open('data_augmentation_info_2.p', 'w'))
