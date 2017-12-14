import numpy as np
import pandas as pd
import cv2
import os

from tsa_reader import read_a3d

data_dir = '/media/penn/DATA2/data'

def get_one_hot_encode(labels_df):
    hash_zone_split = labels_df.Id.str.split('_Zone')
    labels_df.Id = hash_zone_split.str[0]
    labels_df['Zone'] = pd.to_numeric(hash_zone_split.str[1])
    return labels_df.pivot('Id', 'Zone', 'Probability')

def create_image_stack(a3d_fname, save_path='/media/penn/DATA2/data/image_stacks'):
    volume = read_a3d(a3d_fname)
    prefix = os.path.splitext(os.path.basename(a3d_fname))[0]
    stack_dir = os.path.join(save_path, prefix)
    os.mkdir(os.path.join(stack_dir))
    for i in range(volume.shape[0]):
        cv2.imwrite(os.path.join(stack_dir, '{}.tif'.format(i)), convert_scale(volume[i],np.uint16))

def convert_scale(data, dtype):
    info = np.iinfo(dtype)
    return np.interp(data, [data.min(), data.max()], [info.min, info.max]).astype(dtype)


if __name__ == "__main__":
    pass
    csv_path = '/media/penn/DATA2/data/stage1_labels.csv'
    df = get_one_hot_encode(pd.read_csv(csv_path))


