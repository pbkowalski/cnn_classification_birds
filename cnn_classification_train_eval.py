""" Auto Encoder Example.
Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import
from cnn_classification_shared import _parse_folder, _parse_function, parse_file_list, model_path, cnn_model,\
    random_seed, train


import tensorflow as tf

import random

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 16
image_dir = "img3"
#Read images
file_list = _parse_folder(image_dir)
random.seed(random_seed)
random.shuffle(file_list)
labelList = list(map(parse_file_list, file_list))
labels = list(set(labelList))
d = dict(zip(labels, range(0,len(labels))))
labelnList = list(map(lambda x : d[x], labelList))

#split into training and test
filelist_train = file_list[:round(train*len(file_list))]
filelist_test = file_list[round(train*len(file_list)):]

labelList_train = labelnList[:round(train*len(file_list))]
labelList_test = labelnList[round(train*len(file_list)):]

# Create the Estimator
birdsclassifier = tf.estimator.Estimator(
    model_fn=cnn_model, model_dir=model_path)
 # Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=50)
print('begin training')
# Train the model


def input_fn():
    filenamesTrain = tf.constant(filelist_train)
    labelsTrain = tf.constant(labelList_train)
    datasetTrain = tf.data.Dataset.from_tensor_slices((filenamesTrain, labelsTrain))
    datasetTrain = datasetTrain.map(_parse_function)
    datasetTrain = datasetTrain.repeat()  # repeat the input indefinitely;
    datasetTrain = datasetTrain.batch(batch_size)
    iteratorTrain = datasetTrain.make_one_shot_iterator()
    features, labels = iteratorTrain.get_next()
    return {'x': features}, labels
birdsclassifier.train(
    input_fn=input_fn,
    #steps=num_steps,
    steps=500,
    hooks=[logging_hook])
# Evaluate the model and print results
print('begin eval')


def eval_input_fn():
    filenamesTest = tf.constant(filelist_test)
    labelsTest = tf.constant(labelList_test)
    datasetTest = tf.data.Dataset.from_tensor_slices((filenamesTest, labelsTest))
    datasetTest = datasetTest.map(_parse_function)
    datasetTest = datasetTest.batch(1)
    iteratorTest = datasetTest.make_one_shot_iterator()
    featuresTest, labelsT= iteratorTest.get_next()
    return {'x': featuresTest}, labelsT

eval_results = birdsclassifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
eval_results = birdsclassifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
eval_results = birdsclassifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

# Encode and decode images from test set and visualize their reconstruction.
