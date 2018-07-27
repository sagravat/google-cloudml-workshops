import tensorflow as tf
from tensorflow import data
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma
import tensorflow_transform.coders as tft_coders

from tensorflow_transform.beam import impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

import apache_beam as beam
from apache_beam.io.tfrecordio import ReadFromTFRecord
import argparse

import os
import params
import shutil
import dnn_estimator
import featurizer
import input
import metadata
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument(
      '--run-id',
      help='run-id',
      required=True)

args = parser.parse_args()


TRAIN_SIZE = metadata.TRAIN_SIZE
NUM_EPOCHS = metadata.NUM_EPOCHS
BATCH_SIZE = metadata.BATCH_SIZE
TOTAL_STEPS = (TRAIN_SIZE / BATCH_SIZE) * NUM_EPOCHS
EVAL_EVERY_SEC = metadata.EVAL_EVERY_SEC


MODEL_NAME = 'dnn_estimator'  # 'tree_estimator' | 'dnn_estimator'
model_dir = os.path.join(params.Params.MODELS_DIR, MODEL_NAME)
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
print("Steps per Epoch:", TRAIN_SIZE / BATCH_SIZE)
print("Total Steps:", TOTAL_STEPS)

estimator = dnn_estimator.create_estimator(run_config, metadata.hparams)


def generate_eval_receiver_fn(transform_artefacts_dir):
    transformed_metadata = metadata_io.read_metadata(transform_artefacts_dir + "/transformed_metadata")
    transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

    def _eval_receiver_fn():
        serialized_tf_example = tf.placeholder(
            dtype=tf.string, shape=[None], name='input_example_placeholder')

        receiver_tensors = {'examples': serialized_tf_example}
        transformed_features = tf.parse_example(serialized_tf_example, transformed_feature_spec)

        return tfma.export.EvalInputReceiver(
            features=transformed_features,
            receiver_tensors=receiver_tensors,
            labels=transformed_features[metadata.TARGET_FEATURE_NAME])

    return _eval_receiver_fn


eval_model_dir = model_dir + "/export/evaluate"

shutil.rmtree(eval_model_dir, ignore_errors=True)

tfma.export.export_eval_savedmodel(
    estimator=estimator,
    export_dir_base=eval_model_dir,
    eval_input_receiver_fn=generate_eval_receiver_fn(params.Params.TRANSFORM_ARTEFACTS_DIR)
)

transformed_metadata = metadata_io.read_metadata(
    os.path.join(params.Params.TRANSFORM_ARTEFACTS_DIR, "transformed_metadata"))

transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

# print(transformed_feature_spec)

TRANSFORMED_NUMERIC_FEATURE_NAMES = [
    feature_name
    for feature_name in transformed_feature_spec.keys()
    if feature_name.endswith('_scaled')
]

TRANSFORMED_BUCKETIZED_FEATURE_NAMES = [
    feature_name
    for feature_name in transformed_feature_spec.keys()
    if feature_name.endswith('_bucketized')
]

TRANSFORMED_CATEGORICAL_FEATURE_NAMES = metadata.CATEGORICAL_FEATURE_NAMES

TRANSFORMED_INTEGERIZED_CATEGORICAL_FEATURE_NAMES = [
    feature_name
    for feature_name in transformed_feature_spec.keys()
    if feature_name.endswith('_integerized')
]

slice_spec = [tfma.SingleSliceSpec()]
for feature_name in TRANSFORMED_NUMERIC_FEATURE_NAMES + TRANSFORMED_BUCKETIZED_FEATURE_NAMES + TRANSFORMED_CATEGORICAL_FEATURE_NAMES:
    slice_spec += [tfma.SingleSliceSpec(columns=[feature_name])]

# print slice_spec

model_location = os.path.join(eval_model_dir, os.listdir(eval_model_dir)[0])
data_location = params.Params.TRANSFORMED_EVAL_DATA_FILE_PREFIX + "*.tfrecords"


def run_tfma(slice_spec, input_csv, add_metrics_callbacks=None):
    """A simple wrapper function that runs tfma locally.

    A function that does extra transformations on the data and then run model analysis.

    Args:
        slice_spec: The slicing spec for how to slice the data.
        tf_run_id: An id to contruct the model directories with.
        tfma_run_id: An id to construct output directories with.
        input_csv: The evaluation data in csv format.
        add_metrics_callback: Optional list of callbacks for computing extra metrics.

    Returns:
        An EvalResult that can be used with TFMA visualization functions.
    """
    #EVAL_MODEL_DIR = 'eval'
    #eval_model_base_dir = os.path.join(params.Params.MODELS_DIR, EVAL_MODEL_DIR)
    my_eval_model_dir = os.path.join(eval_model_dir, next(os.walk(eval_model_dir))[1][0])
    print(my_eval_model_dir)

    tfma_out = os.path.join(params.Params.TFMA_OUT, args.run_id)
    display_only_data_location = input_csv
    with beam.Pipeline() as pipeline:
        result = (pipeline
            | 'ReadFromTFRecords' >> ReadFromTFRecord(
                params.Params.TRANSFORMED_EVAL_DATA_FILE_PREFIX + '-*')
            | 'EvaluateAndWriteResults' >> tfma.EvaluateAndWriteResults(
                eval_saved_model_path=my_eval_model_dir,
                slice_spec=slice_spec,
                output_path=tfma_out,
                display_only_data_location=input_csv)
        )


    return None #tfma.load_eval_result(output_path=params.Params.TFMA_OUT)

"""
eval_result = tfma.run_model_analysis(
    model_location=model_location ,
    data_location=data_location,
    file_format='tfrecords',
    slice_spec=slice_spec,
#     example_weight_key=None,
#     output_path=None
)
"""

# An empty slice spec means the overall slice, that is, the whole dataset.
OVERALL_SLICE_SPEC = tfma.SingleSliceSpec()

# Data can be sliced along a feature column
# In this case, data is sliced along feature column trip_start_hour.
FEATURE_COLUMN_SLICE_SPEC = tfma.SingleSliceSpec(columns=[''])

# Data can be sliced by crossing feature columns
# In this case, slices are computed for trip_start_day x trip_start_month.
FEATURE_COLUMN_CROSS_SPEC = tfma.SingleSliceSpec(columns=[''])

# Data can be sliced by crossing feature columns
# In this case, slices are computed for trip_start_day x trip_start_month.
FEATURE_COLUMN_CROSS_SPEC = tfma.SingleSliceSpec(columns=[''])

# Metrics can be computed for a particular feature value.
# In this case, metrics is computed for all data where trip_start_hour is 12.
FEATURE_VALUE_SPEC = tfma.SingleSliceSpec(features=[])

# It is also possible to mix column cross and feature value cross.
# In this case, data where trip_start_hour is 12 will be sliced by trip_start_day.
COLUMN_CROSS_VALUE_SPEC = tfma.SingleSliceSpec(columns=[''], features=[])

ALL_SPECS = [
    OVERALL_SLICE_SPEC,
    FEATURE_COLUMN_SLICE_SPEC,
    FEATURE_COLUMN_CROSS_SPEC

]

tfma_result = run_tfma(ALL_SPECS, input_csv=params.Params.RAW_EVAL_DATA_FILE)
