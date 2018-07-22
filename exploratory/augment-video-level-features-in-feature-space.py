import collections
import cPickle as pickle
import glob
import operator
import os
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def grouper(iterable, n=1):
    ''' make iterable groups (each group has n elements) '''
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
    '''
    Find k nearest neighbors

    Input features: dictionary of (k=id, v=d-dim features)
          curr: query id
          top_k: return this number of nearest neighbors
    Returns top_k nearest neighbor tuples (id, distance) in the order of
            increasing distance
    '''

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
    # implement basic ideas from: https://arxiv.org/abs/1702.05538

    # all record files to process
    records = sorted(glob.glob("/media/6TB/video/yt8m-v2/video/train???[13579].tfrecord"))

    # new directory to store all augmented tfrecords
    new_video_feature_dir = "/media/6TB/video/yt8m-v2/video-augmented"
    overwrite = True

    # pre-processed sub/over-sampling information per video_id
    # generated by ./explore_data_aug.py
    tf.logging.info("Reading sub/over-sampling information...")
    vid_label_sampling = pickle.load(open('data_augmentation_info.p', 'r'))
    oversampling_labels = pickle.load(open('data_augmentation_info_2.p', 'r'))

    # inter/extra-polation ratios as in the paper above
    lambda_interp = 0.5
    lambda_extrap = 0.5

    # noise
    noise_sigma = 0.03

    # ------------------------------------------------------------------
    # read this many tfrecords for processing
    # important to read many tfrecords to find good nearest neighbors for
    # classes with fewer samples, yet need to be small enough to meet
    # memory/speed constraints
    num_record = 256
    # ------------------------------------------------------------------

    # for book keeping / diagnostic purpose
    interpolation_count = 0
    add_noise_count = 0
    original_sample_count = 0
    final_sample_count = 0

    for group_count, record_group in enumerate(grouper(records, n=num_record)):
        tf.logging.info("Reading record group ({}/{})...".format(
                    group_count + 1, len(records) / num_record + 1))
        vid_ids = list()
        labels = dict()
        vfeats = dict()
        afeats = dict()
        vfeats_by_class = collections.defaultdict(dict)
        afeats_by_class = collections.defaultdict(dict)
        record_names = dict()
	min_count_labels = set()

        # read all the tfrecord files in this group
        # store all features/labels
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
                original_sample_count += 1

                for l in label:
                    # populate feature dictionary (key=label and video id)
                    # only if this label has few samples (oversampling regime)
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
                # sub-sampling regime and this video was selected
                if vid_label_sampling[vid_id]['selected_if_subsampling'] == True:
                    new_vid_id = vid_id + '_s00'
                    new_vid_ids.append(new_vid_id)
                    new_labels[new_vid_id] = labels[vid_id]
                    new_vfeats[new_vid_id] = vfeats[vid_id]
                    new_afeats[new_vid_id] = afeats[vid_id]
                    new_record_names[new_vid_id] = record_names[vid_id]
                # this video is skipped
                else:
                    continue

            # over-sampling
            elif vid_label_sampling[vid_id]['regime'] == 'oversampling':
                num_aug = int(vid_label_sampling[vid_id]['ratio'])
                label_to_augment = vid_label_sampling[vid_id]['min_count_label']
                nearest_neighbors = find_knn(
                        vfeats_by_class[label_to_augment],
                        vid_id,
                        top_k=num_aug)

                # unchanged/original feature
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
                    # we don't use distance for now

                    # pass if already interpolated
                    if tuple(sorted((vid_id, neighbor))) in interpolated_pairs:
                        continue

                    # interpolate
                    interpolated_vfeats = interpolate(
                            f1=vfeats[vid_id],
                            f2=vfeats_by_class[label_to_augment][neighbor],
                            l=lambda_interp)

                    # although a nearest neighbor is based on video feature,
                    # we interpolate audio features as well, hoping this is not
                    # a bad thing.
                    interpolated_afeats = interpolate(
                            f1=afeats[vid_id],
                            f2=afeats_by_class[label_to_augment][neighbor],
                            l=lambda_interp)

                    # populate features, etc
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

                    # mark this pair "already used in augmentation", so don't
                    # repeat it later
                    interpolated_pairs.add(tuple(sorted((vid_id, neighbor))))
                    interpolation_count += 1

                # if no (valid) nearest neighbor was found :(
                if nearest_neighbors is None or oversample_count == 1:
                    noisy_vfeats = add_noise(vfeats[vid_id], sigma=noise_sigma)
                    noisy_afeats = add_noise(afeats[vid_id], sigma=noise_sigma)

                    new_vid_id = vid_id + '_o{:02d}'.format(oversample_count)
                    new_vid_ids.append(new_vid_id)
                    new_labels[new_vid_id] = labels[vid_id]
                    new_vfeats[new_vid_id] = noisy_vfeats
                    new_afeats[new_vid_id] = noisy_afeats
                    new_record_names[new_vid_id] = record_names[vid_id]

                    oversample_count += 1
                    add_noise_count += 1

            else:
                assert False, "'regime' must be either subsampling or oversampling. Check pre-processina!g"

        # now that we have all sub/over-sampled examples, write to one tfrecord
        # at a time
        for record in record_group:
            record_path, record_file = os.path.split(record)
            new_record_file = os.path.join(new_video_feature_dir, 'AUG_' + record_file)

            if os.path.isfile(new_record_file) and not overwrite:
                tf.logging.warn("new_record_file={} already exists. skipping...".format(new_record_file))
                continue

            tf.logging.info("Writing new tfrecord={}...".format(new_record_file))
            writer = tf.python_io.TFRecordWriter(new_record_file)
            for vid_id, this_record in new_record_names.iteritems():
                # search for videos from this original tfrecord file
                if this_record == record:

                    example = tf.train.Example(features=tf.train.Features(feature={
                            'id': _bytes_feature([tf.compat.as_bytes(vid_id, encoding='utf-8')]),
                            'labels': _int64_feature(new_labels[vid_id]),
                            'mean_rgb': _float_feature(new_vfeats[vid_id]),
                            'mean_audio': _float_feature(new_afeats[vid_id]),
                            }))
                    writer.write(example.SerializeToString())
                    final_sample_count += 1

            writer.close()

    tf.logging.info("-" * 39)
    tf.logging.info("# interpolated samples={}".format(interpolation_count))
    tf.logging.info("# add-noise samples={}".format(add_noise_count))
    tf.logging.info("-" * 19)
    tf.logging.info("BEFORE SUB/OVER-SAMPLED: # of samples={}".format(original_sample_count))
    tf.logging.info("AFTER SUB/OVER-SAMPLED: # of samples={}".format(final_sample_count))
    tf.logging.info("-" * 39)
