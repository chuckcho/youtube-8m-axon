import collections
import cPickle as pickle
import glob
import operator
import os
import pandas as pd
import matplotlib
matplotlib.use('agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def grouper(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def _bytes_feature(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=v))

def _int64_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def _float_feature(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))

def interpolate(f1, f2, l=0.5):
    if type(f1) != np.ndarray or type(f1) != np.ndarray:
        f1 = np.asarray(f1)
        f2 = np.asarray(f2)
    return f1 + (f2 - f1) * l

def extrapolate(f1, f2, l=0.5):
    if type(f1) != np.ndarray or type(f1) != np.ndarray:
        f1 = np.asarray(f1)
        f2 = np.asarray(f2)
    return f1 + (f1 - f2) * l

def add_noise(f, sigma=0.01):
    if type(f) != np.ndarray:
        f = np.asarray(f)
    return f + np.random.normal(0.0, sigma, f.shape)

def l2_distance(f1, f2):
    if type(f1) != np.ndarray or type(f1) != np.ndarray:
        f1 = np.asarray(f1)
        f2 = np.asarray(f2)
    return np.linalg.norm(f1 - f2)

def find_knn(features, curr, top_k=1):
    query = features[curr]
    distances = dict()
    for index in features:
        if index == curr:
            continue
        distances[index] = l2_distance(query, features[index])
    nearest_neighbors = sorted(distances.items(), key=operator.itemgetter(1))[:top_k]

    return nearest_neighbors

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # augmentation

    records = sorted(glob.glob("/media/6TB/video/yt8m-v2/video/train??0[13579].tfrecord"))
    new_video_feature_dir = "/media/6TB/video/yt8m-v2/video-augmented"
    num_classes = 3862
    overwrite = True

    oversampling_labels = pickle.load(open('data_augmentation_info_2.p', 'r'))
    vid_label_sampling = pickle.load(open('data_augmentation_info.p', 'r'))

    lambda_interp = 0.5
    lambda_extrap = 0.5

    # ------------------------------------------------------------------
    # read this many tfrecords for processing
    # important to read many tfrecords to find good nearest neighbors for
    # classes with fewer samples, yet need to be small enough for memory/speed
    # constraint
    num_record = 64
    # ------------------------------------------------------------------

    '''
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

    for record_group in grouper(records, n=num_record):
        vid_ids = list()
        labels = dict()
        vfeats = dict()
        afeats = dict()
        vfeats_by_class = collections.defaultdict(dict)
        afeats_by_class = collections.defaultdict(dict)
        record_names = dict()
	min_count_labels = set()
        for record in record_group:

            for example in tf.python_io.tf_record_iterator(record):
                tf_example = tf.train.Example.FromString(example)
                vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
                label = tf_example.features.feature['labels'].int64_list.value
                vfeat = tf_example.features.feature['mean_rgb'].float_list.value
                afeat = tf_example.features.feature['mean_audio'].float_list.value
                assert vid_id in vid_label_sampling, "vid={} not in training_label.csv. hmmm...".format(vid_id)

                vid_ids.append(vid_id)
                labels[vid_id] = label
                vfeats[vid_id] = vfeat
                afeats[vid_id] = afeat
                record_names[vid_id] = record
                vfeats[vid_id] = vfeat

                for l in label:
                    if l in oversampling_labels:
                        vfeats_by_class[l][vid_id] = vfeat
                        afeats_by_class[l][vid_id] = afeat

        # ------------------------------------------------------------------
        # sub/over-sampling logic
        new_vid_ids = []
        new_labels = dict()
        new_vfeats = dict()
        new_afeats = dict()
        new_record_names = dict()
        interpolated_pairs = set()

        for vid_id in vid_ids:

            # sub-sampling
            if vid_label_sampling[vid_id]['regime'] == 'subsampling':
                if vid_label_sampling[vid_id]['selected_if_subsampling'] == True:
                    new_vid_id = vid_id + '_s00'
                    new_vid_ids.append(new_vid_id)
                    new_labels[new_vid_id] = labels[vid_id]
                    new_vfeats[new_vid_id] = vfeats[vid_id]
                    new_afeats[new_vid_id] = afeats[vid_id]
                    new_record_names[new_vid_id] = record_names[vid_id]
                else:
                    continue

            # over-sampling
            elif vid_label_sampling[vid_id]['regime'] == 'oversampling':
                num_aug = int(vid_label_sampling[vid_id]['ratio'])
                label_to_augment = vid_label_sampling[vid_id]['min_count_label']
                num_nearest_neighbors = num_aug / 2
                nearest_neighbors = find_knn(
                        vfeats_by_class[label_to_augment],
                        vid_id,
                        top_k=num_nearest_neighbors)
                #print "-"*29
                #print "label={}, top_k={}, nearest_neighbors={}".format(label_to_augment, num_nearest_neighbors, nearest_neighbors)

                # unchanged/original
                new_vid_id = vid_id + '_o00'
                new_vid_ids.append(new_vid_id)
                new_labels[new_vid_id] = labels[vid_id]
                new_vfeats[new_vid_id] = vfeats[vid_id]
                new_afeats[new_vid_id] = afeats[vid_id]
                new_record_names[new_vid_id] = record_names[vid_id]

                # if enough nearest_neighbors were found, do interpolation and
                # extrapolation
                oversample_count = 1
                for neighbor, distance in nearest_neighbors:

                    # pass if already interpolated
                    if tuple(sorted((vid_id, neighbor))) in interpolated_pairs:
                        continue

                    # interpolate
                    interpolated_vfeats = interpolate(
                            f1=vfeats[vid_id],
                            f2=vfeats_by_class[label_to_augment][neighbor],
                            l=lambda_interp)
                    interpolated_afeats = interpolate(
                            f1=afeats[vid_id],
                            f2=afeats_by_class[label_to_augment][neighbor],
                            l=lambda_interp)

                    new_vid_id = vid_id + '_o{:02d}'.format(oversample_count)
                    new_vid_ids.append(new_vid_id)
                    new_labels[new_vid_id] = labels[vid_id]
                    new_vfeats[new_vid_id] = interpolated_vfeats
                    new_afeats[new_vid_id] = interpolated_afeats
                    new_record_names[new_vid_id] = record_names[vid_id]

                    oversample_count += 1

                    # extrapolate
                    extrapolated_vfeats = extrapolate(
                            f1=vfeats[vid_id],
                            f2=vfeats_by_class[label_to_augment][neighbor],
                            l=lambda_extrap)
                    extrapolated_afeats = extrapolate(
                            f1=afeats[vid_id],
                            f2=afeats_by_class[label_to_augment][neighbor],
                            l=lambda_extrap)

                    new_vid_id = vid_id + '_o{:02d}'.format(oversample_count)
                    new_vid_ids.append(new_vid_id)
                    new_labels[new_vid_id] = labels[vid_id]
                    new_vfeats[new_vid_id] = extrapolated_vfeats
                    new_afeats[new_vid_id] = extrapolated_afeats
                    new_record_names[new_vid_id] = record_names[vid_id]

                    oversample_count += 1

                    # mark this pair "already used in augmentation"
                    interpolated_pairs.add(tuple(sorted((vid_id, neighbor))))

                # if no (valid) nearest neighbor was found :(
                if nearest_neighbors is None or oversample_count == 1:
                    noisy_vfeats = add_noise(vfeats[vid_id], sigma=0.01)
                    noisy_afeats = add_noise(afeats[vid_id], sigma=0.01)

                    new_vid_id = vid_id + '_o{:02d}'.format(oversample_count)
                    new_vid_ids.append(new_vid_id)
                    new_labels[new_vid_id] = labels[vid_id]
                    new_vfeats[new_vid_id] = noisy_vfeats
                    new_afeats[new_vid_id] = noisy_afeats
                    new_record_names[new_vid_id] = record_names[vid_id]

                    oversample_count += 1

            else:
                assert False, "'regime' must be either subsampling or oversampling. Check pre-processina!g"

        for record in record_group:
            record_path, record_file = os.path.split(record)
            new_record_file = os.path.join(new_video_feature_dir, 'AUG_' + record_file)

            writer = tf.python_io.TFRecordWriter(new_record_file)
            for vid_id, this_record in new_record_names.iteritems():
                if this_record == record:

                    example = tf.train.Example(features=tf.train.Features(feature={
                            'id': _bytes_feature([tf.compat.as_bytes(vid_id, encoding='utf-8')]),
                            'labels': _int64_feature(new_labels[vid_id]),
                            'mean_rgb': _float_feature(new_vfeats[vid_id]),
                            'mean_audio': _float_feature(new_afeats[vid_id]),
                            }))
                    writer.write(example.SerializeToString())

            writer.close()
