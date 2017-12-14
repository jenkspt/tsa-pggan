import tensorflow as tf
import pandas as pd
import numpy as np
from skimage.transform import resize
import os
from scipy import ndimage

from tsa_reader import read_a3d
from util import get_one_hot_encode


MEAN = 4.2554716666666671e-5
STD = 5.0708950000000002e-5
#RESIZE_SHAPE = [256+4, 256+4, 256+2]
RESIZE_SHAPE = [128+2, 128+2, 128+1]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def log_stats(user_id, volume, csv):
    row = [ user_id,
            *ndimage.center_of_mass(volume),
            volume.min(), volume.max(),
            volume.std(),
            volume.mean()
            ]
    csv.write(','.join([str(val) for val in row]) + '\n')
    csv.flush()

data_dir = '/media/penn/DATA2/data'


# Convert labels to multiclass encoded (one-hot-like) vector labels
labels = get_one_hot_encode(pd.read_csv(os.path.join(data_dir, 'stage2_labels.csv')))
print("# Datapoints = ", len(labels))
print('resize shape ', RESIZE_SHAPE)
#train, valid = train_test_split(labels, test_size=.1, stratify=labels.values)
csv = open('data/a3d_file_stats.csv', 'w')
csv.write(','.join(['xm', 'ym', 'zm', 'Min', 'Max', 'Std', 'Mean']) + '\n')

ilocs = np.arange(len(labels))
np.random.shuffle(ilocs)
sample_size = 100
nsteps = len(ilocs)//100
nsteps = int(np.ceil(len(ilocs)/sample_size))

for i in range(nsteps):
    data_labels = labels.iloc[i*sample_size:(i+1)*sample_size]
    tfrecord_fname = os.path.join(data_dir, 'tfrecords/stage2', str(i) + '.tfrecord')
    print("writing", tfrecord_fname)
    writer = tf.python_io.TFRecordWriter(tfrecord_fname)
    fpath = os.path.join(data_dir, 'stage2/a3d', '{}.a3d')

    cnt = 0
    for user_id, vector in data_labels.iterrows():
        volume = read_a3d(fpath.format(user_id))
        log_stats(user_id, volume, csv)
        # Preprocess
        volume = volume[16:-16, 70:-70, :] # Shear off the blank space
        volume = (volume - MEAN)/STD
        volume = resize(volume, RESIZE_SHAPE, 
                order=0, mode='symmetric', preserve_range=True)
        volume = volume.astype(np.float32)
        ax0, ax1, ax2 = volume.shape

        example = tf.train.Example(features=tf.train.Features(feature={
            'ax0': _int64_feature(ax0),
            'ax1': _int64_feature(ax1),
            'ax2': _int64_feature(ax2),
            'label_raw' : _bytes_feature(vector.values.tostring()),
            'volume_raw': _bytes_feature(volume.tostring())
        }))
        cnt += 1
        writer.write(example.SerializeToString())
    print(cnt, 'written')

