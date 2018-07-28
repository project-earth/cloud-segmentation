import os
import tensorflow as tf
from tensorflow.python.client.timeline import Timeline

from learning_lib.nn.cnn import CNN
from learning_lib.nn.autoencoder import convert_to_autoencoder

FILES_PER_BATCH = 2
DATA_DIR = '/home/fwang/earth_data/datasets/cloud-segmentation/autoencoder/data'
MODEL_OUTPUT_DIR = '/home/fwang/earth_data/datasets/cloud-segmentation/autoencoder/models'
PIPELINE_BUFFER_SIZE = 10
SHUFFLE_BUFFER_SIZE = 50
WORKERS = 16
N_EPOCHS = 50
SUMMARY_INTERVAL = 60
CHECKPOINT_INTERVAL = 60 * 5

BATCH_SIZE_PER_FILE = 100 # Cannot be configured, function of TFRecord generation
BATCH_SIZE = FILES_PER_BATCH * BATCH_SIZE_PER_FILE # Cannot be configured, function of TFRecord generation
LC = [
    {
        'layer_type': 'conv',
        'filter_size': [6, 6, 10, 64],
        'init_filter_mean': 0.0,
        'init_filter_stddev': 0.01,
        'init_bias_mean': 0.0,
        'init_bias_stddev': 0.01,
        'stride_size': [1, 2, 2, 1],
        'activation': tf.tanh,
        'output_size': [62, 62, 64]
    },
    {
        'layer_type': 'conv',
        'filter_size': [6, 6, 64, 64],
        'init_filter_mean': 0.0,
        'init_filter_stddev': 0.01,
        'init_bias_mean': 0.0,
        'init_bias_stddev': 0.01,
        'stride_size': [1, 1, 1, 1],
        'activation': tf.tanh,
        'output_size': [57, 57, 64]
    },
    {
        'layer_type': 'conv',
        'filter_size': [4, 4, 64, 64],
        'init_filter_mean': 0.0,
        'init_filter_stddev': 0.01,
        'init_bias_mean': 0.0,
        'init_bias_stddev': 0.01,
        'stride_size': [1, 1, 1, 1],
        'activation': tf.tanh,
        'output_size': [54, 54, 64]
    },
    {
        'layer_type': 'conv',
        'filter_size': [4, 4, 64, 64],
        'init_filter_mean': 0.0,
        'init_filter_stddev': 0.01,
        'init_bias_mean': 0.0,
        'init_bias_stddev': 0.01,
        'stride_size': [1, 1, 1, 1],
        'activation': tf.tanh,
        'output_size': [51, 51, 64]
    },
    {
        'layer_type': 'conv_transpose',
        'filter_size': [4, 4, 64, 64],
        'init_filter_mean': 0.0,
        'init_filter_stddev': 0.01,
        'init_bias_mean': 0.0,
        'init_bias_stddev': 0.01,
        'stride_size': [1, 1, 1, 1],
        'activation': tf.tanh,
        'output_size': [BATCH_SIZE, 54, 54, 64]
    },
    {
        'layer_type': 'conv_transpose',
        'filter_size': [4, 4, 64, 64],
        'init_filter_mean': 0.0,
        'init_filter_stddev': 0.01,
        'init_bias_mean': 0.0,
        'init_bias_stddev': 0.01,
        'stride_size': [1, 1, 1, 1],
        'activation': tf.tanh,
        'output_size': [BATCH_SIZE, 57, 57, 64]
    },
    {
    'init_filter_stddev': 0.01,
        'layer_type': 'conv_transpose',
        'filter_size': [6, 6, 64, 64],
        'init_filter_mean': 0.0,
        'init_bias_mean': 0.0,
        'init_bias_stddev': 0.01,
        'stride_size': [1, 1, 1, 1],
        'activation': tf.tanh,
        'output_size': [BATCH_SIZE, 62, 62, 64]
    },
    {
        'layer_type': 'conv_transpose',
        'filter_size': [6, 6, 10, 64],
        'init_filter_mean': 0.0,
        'init_filter_stddev': 0.01,
        'init_bias_mean': 0.0,
        'init_bias_stddev': 0.01,
        'stride_size': [1, 2, 2, 1],
        'activation': tf.tanh,
        'output_size': [BATCH_SIZE, 128, 128, 10]
    }
]


# ================= Define Input Pipeline ================= #
def decode_image(proto):
    features = {
        'grid_size': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)
    }

    parsed_features = tf.parse_single_example(proto, features)
    grid_size = tf.cast(parsed_features['grid_size'], tf.int64)
    channels = tf.cast(parsed_features['channels'], tf.int64)
    image = tf.decode_raw(parsed_features['image'], tf.float32)
    image = image / (2 ** 16) - 0.5
    return tf.reshape(image, tf.stack([-1, grid_size, grid_size, channels]))

def merge_batch(proto):
    proto = tf.reshape(proto, (-1, 128, 128, 10))
    return tf.random_shuffle(proto)

def define_input_pipe():
    file_list = [os.path.join(DATA_DIR, filename) for filename in os.listdir(DATA_DIR)]
    ds = tf.data.TFRecordDataset(file_list)
    ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
    ds = ds.map(decode_image, num_parallel_calls=WORKERS)
    ds = ds.batch(FILES_PER_BATCH)
    ds = ds.map(merge_batch, num_parallel_calls=WORKERS)
    ds = ds.prefetch(buffer_size=PIPELINE_BUFFER_SIZE)
    return ds.make_initializable_iterator()


# ================= Define Network ================= #
def define_network(iterator):
    pipeline = iterator.get_next()
    cnn = CNN(
        LC, optimizer=tf.train.AdamOptimizer(),
        input_vector=pipeline,
        train_targets_vector=pipeline,
        logdir=MODEL_OUTPUT_DIR,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        summary_interval=SUMMARY_INTERVAL
    )
    cnn = convert_to_autoencoder(cnn, 3)
    return cnn


# ================= Add Tensorboard Metrics ================= #
def define_tensorboard_metrics(network):
    # Loss Function
    tf.summary.scalar('Training Loss', network.loss_val)

    # Input Image
    for i in range(10):
        tf.summary.image('Input B{}'.format(i), network.input[:,:,:,i:(i+1)])

    # Output Image
    for i in range(10):
        tf.summary.image('Output B{}'.format(i), network.output[:,:,:,i:(i+1)])


# ================= Main ================= #
def main():
    # run_metadata = tf.RunMetadata()
    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    pipeline_iterator = define_input_pipe()
    print("Input pipeline created")

    cnn = define_network(pipeline_iterator)
    define_tensorboard_metrics(cnn)
    print("CNN object initialized")

    # Scaffold needed to address bug found here:
    # https://github.com/tensorflow/tensorflow/issues/12859
    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), pipeline_iterator.initializer)
    )
    cnn.init_session(managed=True, scaffold=scaffold)

    print("CNN session started, starting training...")
    for epoch in range(N_EPOCHS):
        print("Training epoch {}".format(epoch))
        cnn.session.run(pipeline_iterator.initializer)
        cnn.train_online() # options=options, run_metadata=run_metadata)

    # trace = Timeline(step_stats=run_metadata.step_stats)
    # with open('timeline.ctf.json', 'w') as f:
    #     f.write(trace.generate_chrome_trace_format())
    # print("Wrote profiling results to disk")


if __name__ == '__main__':
    main()
