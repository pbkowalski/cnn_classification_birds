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

import tensorflow as tf
from cnn_classification_shared import _parse_folder, cnn_model, parse_function_nolabel, model_path

kasiosList = _parse_folder("img4")
# Create the Estimator
birdsclassifier = tf.estimator.Estimator(
    model_fn=cnn_model, model_dir=model_path)
 # Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=50)

print('begin predict')
def predict_input_fn():
    kasiosFilenames = tf.constant(kasiosList)
    datasetKasios = tf.data.Dataset.from_tensor_slices(kasiosFilenames)
    datasetKasios = datasetKasios.map(parse_function_nolabel)
    datasetKasios = datasetKasios.batch(1)
    kasiosIt = datasetKasios.make_one_shot_iterator()
    featuresKasios = kasiosIt.get_next()
    return {'x': featuresKasios}


predict_results = birdsclassifier.predict(input_fn=predict_input_fn)
print(list(predict_results))