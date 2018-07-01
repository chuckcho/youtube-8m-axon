import glob
import os
import pandas as pd
import tensorflow as tf

labels_df = pd.read_csv(os.path.join('..', 'label_names_2018_fixed.csv'))
print(labels_df.head())
print("Total nubers of labels in sample dataset: %s" %(len(labels_df['label_name'].unique())))

video_data_path = '/media/6TB/video/yt8m-v2/video'
video_files = glob.glob(os.path.join(video_data_path, "train????.tfrecord"))

# let's just use the first sample
video_files = [video_files[0]]
print("Will take a look at {}...".format(video_files))

for video_file in video_files:
    for example in tf.python_io.tf_record_iterator(video_file):
        tf_example = tf.train.Example.FromString(example)
        vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
        labels = tf_example.features.feature['labels'].int64_list.value

        # Lets convert labels for each video into their respective names
        label_names = []
        for label_id in labels:
            # some labels ids are missing so have put try/except
            try:
                label_names.append(str(labels_df[labels_df['label_id']==label_id]['label_name'].values[0]))
            except:
                continue
        print('Check video_id at https://data.yt8m.org/2/j/i/{}/{}.js : labels={}'.format(vid_id[:2], vid_id, str(label_names)))
