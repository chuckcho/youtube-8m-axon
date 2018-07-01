import glob
import os
import pandas as pd
import tensorflow as tf

# whether print label names or label ids (integer)
print_label_names = False

video_data_path = '/media/6TB/video/yt8m-v2/video'
#video_files = glob.glob(os.path.join(video_data_path, "validate???5.tfrecord"))
video_files = glob.glob(os.path.join(video_data_path, "train???[13579].tfrecord"))

if print_label_names:
    labels_df = pd.read_csv(os.path.join('..', 'label_names_2018_fixed.csv'))
    #print(labels_df.head())
    #with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    #    print(labels_df)

for video_file in video_files:
    for example in tf.python_io.tf_record_iterator(video_file):
        tf_example = tf.train.Example.FromString(example)
        vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
        labels = tf_example.features.feature['labels'].int64_list.value

        # Lets convert labels for each video into their respective names
        if print_label_names:
            label_names = []
            for label_id in labels:
                # some labels ids are missing so have put try/except
                try:
                    label_names.append(str(labels_df[labels_df['label_id']==label_id]['label_name'].values[0]))
                except:
                    continue
            #print('Check video_id at https://data.yt8m.org/2/j/i/{}/{}.js : labels={}'.format(vid_id[:2], vid_id, str(label_names)))
            #print('{}: {}'.format(vid_id, labels))
            print('{}: {}'.format(vid_id, str(label_names)))
        else:
            print('{}: {}'.format(vid_id, ' '.join(str(x) for x in labels)))
