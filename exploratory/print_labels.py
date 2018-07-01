import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os

data_path = '/media/6TB/video/yt8m-v2'
video_data_path = os.path.join(data_path, 'video')

labels_df = pd.read_csv(os.path.join('..', 'label_names_2018_fixed.csv'))
print(labels_df.head())
print("Total nubers of labels in sample dataset: %s" %(len(labels_df['label_name'].unique())))

video_files = [os.path.join(video_data_path, vf) for vf in os.listdir(video_data_path)]

# let's just use the first sample
video_files = [video_files[0]]
print("Will take a look at {}...".format(video_files))

vid_ids = []
labels = []

for file in video_files:
    for example in tf.python_io.tf_record_iterator(file):
        tf_example = tf.train.Example.FromString(example)

        vid_ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
        labels.append(tf_example.features.feature['labels'].int64_list.value)

print('Number of videos in Sample data set: %s' % str(len(vid_ids)))

for video_index in range(len(vid_ids)):

    vid_id = vid_ids[video_index]

    # Lets convert labels for each video into their respective names
    labels_name = []
    for row in labels:
        n_labels = []
        for label_id in row:
            # some labels ids are missing so have put try/except
            try:
                n_labels.append(str(labels_df[labels_df['label_id']==label_id]['label_name'].values[0]))
            except:
                continue
        labels_name.append(n_labels)

    print('Check video_id at https://data.yt8m.org/2/j/i/{}/{}.js : labels={}'.format(vid_id[:2], vid_id, str(labels_name[video_index])))
