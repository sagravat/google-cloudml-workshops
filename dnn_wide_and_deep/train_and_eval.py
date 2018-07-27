
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
import tensorflow.contrib.eager as tfe

# tf.enable_eager_execution()


import apache_beam as beam

import os
import shutil
import dnn_estimator
import featurizer
import metadata
from metadata import Params
import input
from datetime import datetime
import confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
TRAIN_SIZE = metadata.TRAIN_SIZE
NUM_EPOCHS = metadata.NUM_EPOCHS
BATCH_SIZE = metadata.BATCH_SIZE
TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS
EVAL_EVERY_SEC = metadata.EVAL_EVERY_SEC


params = Params()

parser = argparse.ArgumentParser()
parser.add_argument(
      '--run-id',
      help='run-id',
      required=True)

parser.add_argument(
      '--transform-train-prefix',
      help='transform train file prefix',
      required=True)

parser.add_argument(
      '--transform-eval-prefix',
      help='transform eval file prefix',
      required=True)

args = parser.parse_args()

MODEL_NAME = 'dnn_estimator'
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


def generate_json_serving_fn():

    # get the feature_spec of raw data
    raw_metadata = featurizer.create_raw_metadata()
    raw_placeholder_spec = raw_metadata.schema.as_batched_placeholders()
    raw_placeholder_spec.pop(metadata.TARGET_FEATURE_NAME)

    def _serving_fn():

        raw_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(raw_placeholder_spec)
        raw_features, recevier_tensors, _ = raw_input_fn()

        # apply tranform_fn on raw features
        _, transformed_features = (
            saved_transform_io.partially_apply_saved_transform(
                os.path.join(params.TRANSFORM_ARTIFACTS_DIR, transform_fn_io.TRANSFORM_FN_DIR),
            raw_features)
        )

        # apply the process_features function to transformed features
        transformed_features = input.process_features(transformed_features)

        return tf.estimator.export.ServingInputReceiver(
            transformed_features, raw_features)

    return _serving_fn


confusionMatrixSaveHook = confusion_matrix.SaverHook(
    labels = ['0', '1'],
    confusion_matrix_tensor_name = 'mean_iou/total_confusion_matrix',
    summary_writer = tf.summary.FileWriterCache.get(model_dir + "/eval")
)

train_spec = tf.estimator.TrainSpec(
    input_fn = input.generate_tfrecords_input_fn(
        args.transform_train_prefix + "*",
        mode = tf.estimator.ModeKeys.TRAIN,
        num_epochs=metadata.hparams.num_epochs,
        batch_size=metadata.hparams.batch_size
    ),
    max_steps=metadata.hparams.max_steps,
    hooks=None
)

eval_spec = tf.estimator.EvalSpec(
    input_fn = input.generate_tfrecords_input_fn(
        args.transform_eval_prefix + "*",
        mode=tf.estimator.ModeKeys.EVAL,
        num_epochs=1,
        batch_size=metadata.hparams.batch_size
    ),
    exporters=[tf.estimator.LatestExporter(
        name="estimate", # the name of the folder in which the model will be exported to under export
        serving_input_receiver_fn=generate_json_serving_fn(),
        exports_to_keep=1,
        as_text=False)],
    steps=None,
    # hooks=[confusionMatrixSaveHook],
    throttle_secs=EVAL_EVERY_SEC
)


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.TARGET_LABELS)
    )
    return table.lookup(label_string_tensor)

if params.TRAIN:
    if not params.RESUME_TRAINING:
        print("Removing previous training artefacts...")
        shutil.rmtree(model_dir, ignore_errors=True)
    else:
        print("Resuming training...")


    tf.logging.set_verbosity(tf.logging.INFO)

    time_start = datetime.utcnow()
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")

    estimator = dnn_estimator.create_estimator(run_config, metadata.hparams)
    # estimator = tf.contrib.estimator.add_metrics(estimator,
    #                 lambda labels, predictions: {
    #                 'mean_iou': tf.metrics.mean_iou(labels, predictions['class_ids'], 2)})


    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    time_end = datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
else:
    print "Training was skipped!"
