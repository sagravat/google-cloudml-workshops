import tensorflow as tf
from tensorflow import data
import tensorflow_transform as tft
import tensorflow_transform.coders as tft_coders

from tensorflow_transform.beam import impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
import argparse

import apache_beam as beam

import os
import shutil
import dnn_estimator
import input
import metadata
from metadata import Params
from datetime import datetime

params = Params()
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

TRAIN_SIZE = metadata.TRAIN_SIZE
NUM_EPOCHS = metadata.NUM_EPOCHS
BATCH_SIZE = metadata.BATCH_SIZE
TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS
EVAL_EVERY_SEC = metadata.EVAL_EVERY_SEC

parser = argparse.ArgumentParser()
parser.add_argument(
      '--run-id',
      help='run-id',
      required=True)

parser.add_argument(
      '--transform-test-prefix',
      help='transform test file prefix',
      required=True)

args = parser.parse_args()

MODEL_NAME = 'dnn_estimator' # 'tree_estimator' | 'dnn_estimator'
model_dir = os.path.join(params.MODELS_DIR, MODEL_NAME)
model_dir = os.path.join(model_dir, args.run_id)

run_config = tf.estimator.RunConfig(
    tf_random_seed=19830610,
    log_step_count_steps=1000,
    save_checkpoints_secs=EVAL_EVERY_SEC,
    keep_checkpoint_max=3,
    model_dir=model_dir
)


print(metadata.hparams)
print("")
print("Model Directory:", run_config.model_dir)
print("Dataset Size:", TRAIN_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Steps per Epoch:",TRAIN_SIZE/BATCH_SIZE)
print("Total Steps:", TOTAL_STEPS)


VALID_SIZE = 2775

tf.logging.set_verbosity(tf.logging.ERROR)

estimator = dnn_estimator.create_estimator(run_config, metadata.hparams)


test_metrics = estimator.evaluate(
    input_fn=input.generate_tfrecords_input_fn(
        files_name_pattern= args.transform_test_prefix + "*",
        mode= tf.estimator.ModeKeys.EVAL,
        batch_size= VALID_SIZE),
    steps=1
)
print("")
print("############################################################################################")
print("# Test Measures: {}".format(test_metrics))
print("############################################################################################")

