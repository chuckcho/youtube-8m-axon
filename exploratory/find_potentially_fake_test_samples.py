import tensorflow as tf
import numpy as np
import glob

records = glob.glob("/media/6TB/video/yt8m-v2/video/test????.tfrecord")
potential_fake_vid_id_outfile = './fishy_test_vid_ids.txt'

with open(potential_fake_vid_id_outfile, 'w') as of:
    for record in records:
        for example in tf.python_io.tf_record_iterator(record):
            tf_example = tf.train.Example.FromString(example)

            vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
            std_mean_rgb = np.std(tf_example.features.feature['mean_rgb'].float_list.value)
            looks_fishy = std_mean_rgb < 0.1
            if looks_fishy:
                of.write(vid_id + '\n')
