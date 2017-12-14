
import tensorflow as tf
import os
from glob import glob


def _parse_function(example_proto):
    features = {
            'ax0': tf.FixedLenFeature([], tf.int64),
            'ax1': tf.FixedLenFeature([], tf.int64),
            'ax2': tf.FixedLenFeature([], tf.int64),
            "label_raw": tf.FixedLenFeature([], tf.string, default_value=""),
            "volume_raw": tf.FixedLenFeature([], tf.string, default_value="")
    }
    parsed_features = tf.parse_single_example(example_proto, features)

    ax0 = tf.cast(parsed_features['ax0'], tf.int32)
    ax1 = tf.cast(parsed_features['ax1'], tf.int32)
    ax2 = tf.cast(parsed_features['ax2'], tf.int32)

    label = tf.decode_raw(parsed_features['label_raw'], tf.int64)
    label = tf.cast(label, tf.float32)

    volume = tf.decode_raw(parsed_features['volume_raw'], tf.float32)
    vol_shape = tf.stack([ax0, ax1, ax2])
    volume = tf.reshape(volume, [ax0, ax1, ax2, 1])
    return volume, label 

def _preprocess_train(volume, label):
    #volume = tf.random_crop(volume, [256, 256, 256, 1])
    volume = tf.random_crop(volume, [128, 128, 128, 1])
    volume, label = _random_reverse_axis_zero(volume, label)
    return volume, label

def _random_reverse_axis_zero(volume, label):
    random = tf.random_uniform([], maxval=1, dtype=tf.float32)
    half = tf.constant(.5)
    volume, label = tf.cond(tf.less(random, half), 
            lambda: _reverse_axis_zero(volume, label), 
            lambda: (volume, label))
    return volume, label

def _reverse_axis_zero(volume, label):
    """ Humans are symmetrical, so augment data by reversing
    axis 0, and flipping the correct zones """
    volume = tf.reverse(volume, axis=[0], name='reverse_axis_0')
    label = tf.stack([
        label[2], label[3], label[0], label[1], label[4],
        label[6], label[5], label[9], label[8], label[7],
        label[11], label[10], label[13], label[12], 
        label[15], label[14], label[16]], axis=0)
    return volume, label

def _central_crop(volume):
    #volume = volume[2:-2, 2:-2, 1:-1, :]
    volume = volume[1:-1, 1:-1, :-1, :]
    return volume

def _preprocess_eval(volume, label):
    volume = _central_crop(volume)
    return volume, label


def get_train_fn(filenames, batch_size=1, buffer_size=6):

    def train_data():
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)  # Parse the record into tensors.
        dataset = dataset.map(_preprocess_train)  # Parse the record into tensors.
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat(None)  # Repeat the input indefinitely.
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return train_data

def get_eval_fn(filenames, batch_size=1):
    def eval_data():
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)  # Parse the record into tensors.
        dataset = dataset.map(_preprocess_eval)  # Parse the record into tensors.
        dataset = dataset.repeat(1)  # Run through the eval data once
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return eval_data

