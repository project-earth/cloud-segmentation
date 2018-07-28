import os
import h5py
import MySQLdb as sql
import random
import numpy as np
import tensorflow as tf
import datacollection.settings as settings
from datacollection.utils import generateFilePathStr


bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9', 'B10', 'B11']
grid_size = 128
n_images = 300
batch_size = 100
batches_per_image = 20
dataset_dir = os.path.join(settings.DATA_DIR, 'datasets', 'cloud-segmentation', 'autoencoder', 'data')


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_lids(n_images):
    db = sql.connect(
        host=settings.DB_HOST, user=settings.DB_USER,
        passwd=settings.DB_PASS, db=settings.DB
    )
    cur = db.cursor()
    cur.execute('SELECT lid FROM imageindex order by RAND() LIMIT {};'.format(n_images))
    return [result[0] for result in cur.fetchall()]


def get_points_for_image(image, n, h5F, filename):
    dim = h5F[image][bands[0]].shape
    dataset = np.empty((n, len(bands), grid_size, grid_size))
    writer = tf.python_io.TFRecordWriter(os.path.join(dataset_dir, filename))

    for j in range(n):
        subimage = np.zeros((len(bands), grid_size, grid_size))
        while (subimage == 0).any():
            x = random.randint(grid_size / 2, dim[0] - grid_size / 2 - 1)
            y = random.randint(grid_size / 2, dim[1] - grid_size / 2 - 1)
            for i, b in enumerate(bands):
                subimage[i] = h5F[image][b][
                    int(x - grid_size / 2) : int(x + grid_size / 2),
                    int(y - grid_size / 2) : int(y + grid_size / 2)
                ]
        dataset[j] = subimage

    dataset = dataset.transpose((0, 2, 3, 1))
    dataset = dataset.astype('float32')

    example = tf.train.Example(features=tf.train.Features(feature={
        'grid_size': _int64_feature(grid_size),
        'channels': _int64_feature(len(bands)),
        'image': _bytes_feature(dataset.tostring())
    }))
    writer.write(example.SerializeToString())
    writer.close()


def main():
    images = get_lids(n_images)
    with h5py.File(generateFilePathStr(kind='database'), 'r') as h5F:
        for image in images:
            print('Generating record for {}'.format(image))
            for batch in range(batches_per_image):
                get_points_for_image(image, batch_size, h5F, '{}_{}.tfrecord'.format(image, batch))


if __name__ == '__main__':
    main()
