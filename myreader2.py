import tensorflow as tf
from tensorflow import gfile
import os, fnmatch, sys
import requests
import pandas as pd
import csv
from collections import defaultdict
import numpy as np


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias


def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be
      cast to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized


def get_video_matrix(  features,
                       feature_size,
                       max_frames,
                       max_quantized_value,
                       min_quantized_value):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
      num_frames: number of frames in the sequence
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = Dequantize(decoded_features,
                                      max_quantized_value,
                                      min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames

def prepare_reader(  filename_queue,
                     max_quantized_value=2,
                     min_quantized_value=-2):
    """Creates a single reader thread for YouTube8M SequenceExamples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A tuple of video indexes, video features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={"video_id": tf.FixedLenFeature(
            [], tf.string),
                          "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in feature_names
        })

    # read ground truth labels
    labels = (tf.cast(
        tf.sparse_to_dense(contexts["labels"].values, (num_classes,), 1,
            validate_indices=False),
        tf.bool))

    # loads (potentially) different types of features and concatenates them
    num_features = len(feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    num_frames = -1  # the number of frames in the video
    feature_matrices = [None] * num_features  # an array of different features
    for feature_index in range(num_features):
      feature_matrix, num_frames_in_this_feature = get_video_matrix(
          features[feature_names[feature_index]],
          feature_sizes[feature_index],
          max_frames,
          max_quantized_value,
          min_quantized_value)
      if num_frames == -1:
        num_frames = num_frames_in_this_feature
      else:
        tf.assert_equal(num_frames, num_frames_in_this_feature)

      feature_matrices[feature_index] = feature_matrix

    # cap the number of frames at self.max_frames
    num_frames = tf.minimum(num_frames, max_frames)

    # concatenate different features
    video_matrix = tf.concat(feature_matrices, 1)

    # convert to batch format.
    # TODO: Do proper batch reads to remove the IO bottleneck.
    batch_video_ids = tf.expand_dims(contexts["video_id"], 0)
    batch_video_matrix = tf.expand_dims(video_matrix, 0)
    batch_labels = tf.expand_dims(labels, 0)
    batch_frames = tf.expand_dims(num_frames, 0)

    return batch_video_ids, batch_video_matrix, batch_labels, batch_frames





def write_to_record(id_batch, label_batch, predictions, filenum, num_examples_processed):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_dir + '/' + 'predictions-%04d.tfrecord' % filenum)
    for i in range(num_examples_processed):
        video_id = id_batch[i]
        label = np.nonzero(label_batch[i,:])[0]
        example = get_output_feature(video_id, label, [predictions[i,:]], ['predictions'])
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def get_output_feature(video_id, labels, features, feature_names):
    feature_maps = {'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))}
    for feature_index in range(len(feature_names)):
        feature_maps[feature_names[feature_index]] = tf.train.Feature(
            float_list=tf.train.FloatList(value=features[feature_index]))
    example = tf.train.Example(features=tf.train.Features(feature=feature_maps))
    return example

def get_num_vids(vid_file):
  count = 0
  for example in tf.python_io.tf_record_iterator(vid_file):
    count += 1
  return count





"""
#distill_prediction = pd.read_csv('./baseline-pduan-train_plus_validate-cp_274278.csv', names= ['VideoId', 'LabelConfidencePairs'])
count = 0
distill_dict = defaultdict(list)
with open('./baseline-pduan-train_plus_validate-cp_274278.csv') as fd:

  for i, line in enumerate(fd):
    if i == 0 : continue
    distill_vid_id = line.split(',')[0]
    distill_pred =[]
    l = line.split(',')[1].split('\n')[0].split(' ')
    idx = 0
    for c, s in zip(l[:-1],l[1:]):
      idx += 1 
      if idx % 2 == 0: continue
      distill_pred.append( (int(c), float(s)) )
    distill_dict[distill_vid_id] = distill_pred
    count += 1
    

print('count = {}'.format(count))

sys.exit()
"""

data_path="/media/6TB/video/yt8m-v2/frame"
axon_train_set="./teacher_model_predictions/NetVLADModelLF/*.tfrecord" #"/media/6TB/video/yt8m-v2/frame/train???[13579].tfrecord"
label_mapping = pd.read_csv('./vocabulary.csv', encoding='utf-8',  header=0) #,index_col=0,squeeze=True)#.T.to_dict()
label_mapping_dict = defaultdict(str)
for i, n in zip(label_mapping['Index'], label_mapping['Name']):
    if not n == 'nan' and n: continue 
    label_mapping_dict[i] = str(n).encode('utf-8').decode("utf-8")

video_files = gfile.Glob(axon_train_set)
"""
num_classes=3862
feature_sizes=[1024, 128]
feature_names=["mean_rgb", "mean_audio"]
max_frames=300

num_epochs = get_num_vids(video_files[0])
print(num_epochs)

filename_queue = tf.train.string_input_producer(
        video_files, num_epochs=num_epochs, shuffle=False)
print(filename_queue.get_shape())
batch_video_id, batch_mtx, batch_label, batch_frame = prepare_reader(filename_queue)
print(type(batch_video_id))
tf.Print(batch_video_id, [batch_video_id])

#video_id = np.concatenate(batch_video_id, axis=0)

#video_label = np.concatenate(batch_video_label, axis=0)
#video_features = np.concatenate(batch_video_features, axis=0)
#write_to_record(video_id, video_label, video_features, filenum, num_examples_processed)


sys.exit()
"""

mean_rgb = []
mean_audio = []
labels = []
vid_dict = defaultdict(list)
count = 0
print('--->', video_files[0])
for example in tf.python_io.tf_record_iterator(video_files[0]):
    #if count == 10: break
    count += 1
    tf_example = tf.train.Example.FromString(example)

    rand_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
    labels.append(map(int, tf_example.features.feature['labels'].int64_list.value))
    mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
    mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
    #vid_dict[rand_id] = map(int, tf_example.features.feature['labels'].int64_list.value)
print('Number of videos in this tfrecord: ', count)

#for v, l in vid_dict.items():
#    print('{} : {}'.format(v, l))





