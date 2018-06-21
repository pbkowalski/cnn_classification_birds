import tensorflow as tf
import pandas as pd
import os
#Model parameters
model_path = '/tmp/birds_cnn_model'
filters = 64
random_seed = 1356
train = 0.9 #train-evaluation split


def _parse_folder(image_dir):
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
    extensions = ['jpg']
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.' + extension)
        file_list.extend(tf.gfile.Glob(file_glob))
    if not file_list:
        tf.logging.warning('No files found')
    return file_list


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, ratio = 8, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    return image, label


def parse_function_nolabel(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, ratio = 8, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    return image


def create_dictionary(csv):
    df = pd.read_csv(csv)
    cols = df.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
    df.columns = cols
    return dict(zip(list(df.File_ID), list(df.English_name)))


def parse_file_list(filename):
    label_map = create_dictionary('AllBirdsv4.csv')
    foo = filename.split(os.sep)[-1].split('.')[0]
    val = int(foo)
    return label_map[val]


# Construct model
def cnn_model(features, labels, mode):
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # using 8x downsampling input is n*68*85*1
    conv1 = tf.layers.conv2d(
        inputs=features["x"],
        filters=filters,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu)
    # 68 * 85 * f
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # 34 * 43 * 16
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=2 * filters,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu)
    # 34 * 43 * 16

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)  # 17 * 21 * 8
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=4 * filters,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu)
    # 34 * 43 * 16

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    # 17 * 21 * 8

    num_input = 17 * 21 * (filters * 2)
    x_flat = tf.reshape(pool3, [-1, 20480])
    dense = tf.layers.dense(inputs=x_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=19)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"], name = 'acc_op')
    tf.summary.scalar('accuracy', accuracy[1])
    eval_metric_ops = {"accuracy": accuracy}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

