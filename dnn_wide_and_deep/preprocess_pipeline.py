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

import apache_beam as beam

import os
import featurizer
import metadata
from metadata import Params



params = Params()

NUM_BUCKETS = 4

def preprocess(input_features):

    output_features = {}

    output_features[metadata.TARGET_FEATURE_NAME] = input_features[metadata.TARGET_FEATURE_NAME]

    for feature_name in metadata.NUMERIC_FEATURE_NAMES:

        #output_features[feature_name+"_scaled"] = tft.scale_to_z_score(input_features[feature_name])
        output_features[feature_name] = tft.scale_to_z_score(input_features[feature_name])

        quantiles = tft.quantiles(input_features[feature_name], num_buckets=NUM_BUCKETS, epsilon=0.01)
        output_features[feature_name+"_bucketized"] = tft.apply_buckets(input_features[feature_name],
                                                                        bucket_boundaries=quantiles)

    for feature_name in metadata.CATEGORICAL_FEATURE_NAMES:

        tft.uniques(input_features[feature_name], vocab_filename=feature_name)
        output_features[feature_name] = input_features[feature_name]

        # sba added this
        #output_features[feature_name+"_integerized"] = tft.string_to_int(input_features[feature_name],
                                                           #vocab_filename=feature_name)
    for feature_name in metadata.VOCAB_FEATURE_NAMES:

        output_features[feature_name +"_integerized"] = tft.string_to_int(input_features[feature_name],top_k=metadata.VOCAB_SIZE, num_oov_buckets=metadata.OOV_SIZE, vocab_filename=feature_name)
                                                           


    return output_features

class MapAndFilterErrors(beam.PTransform):
  """Like beam.Map but filters out erros in the map_fn."""

  class _MapAndFilterErrorsDoFn(beam.DoFn):
    """Count the bad examples using a beam metric."""

    def __init__(self, fn):
      self._fn = fn
      # Create a counter to measure number of bad elements.
      self._bad_elements_counter = beam.metrics.Metrics.counter(
          'my_example', 'bad_elements')

    def process(self, element):
      try:
        yield self._fn(element)
      except Exception:  # pylint: disable=broad-except
        # Catch any exception the above call.
        self._bad_elements_counter.inc(1)

  def __init__(self, fn):
    self._fn = fn

  def expand(self, pcoll):
    return pcoll | beam.ParDo(self._MapAndFilterErrorsDoFn(self._fn))


def fix_comma_and_filter_third_column(line):
    # to avoid namespace error with DataflowRunner the import of csv is done
    # locacally see https://cloud.google.com/dataflow/faq#how-do-i-handle-nameerrors
    import csv
    cols = list(csv.reader([line], skipinitialspace=True,))[0]
    #return ','.join(cols[0:2] + cols[3:])
    return '\t'.join(cols[1:4] + cols[8:24] + cols[25:39])

def run_transformation_pipeline(args, options):

    options = beam.pipeline.PipelineOptions(flags=[], **options)

    print("Source raw train data files: {}".format(args.raw_train_file))
    print("Source raw train data files: {}".format(args.raw_eval_file))

    print("Sink transformed train data files: {}".format(args.transform_train_prefix))
    print("Sink transformed data files: {}".format(args.transform_eval_prefix))
    print("Sink transform artefacts directory: {}".format(params.TRANSFORM_ARTIFACTS_DIR))

    print("Temporary directory: {}".format(params.TEMP_DIR))
    print("")


    with beam.Pipeline(runner, options=options) as pipeline:
        with impl.Context(params.TEMP_DIR):

            raw_metadata = featurizer.create_raw_metadata()
            converter = tft_coders.csv_coder.CsvCoder(column_names=metadata.RAW_FEATURE_NAMES,
                                                      delimiter=params.RAW_DATA_DELIMITER,
                                                      schema=raw_metadata.schema)

            ###### analyze & transform train #########################################################
            if(runner=='DirectRunner'):
                print("Transform training data....")

            step = 'train'

            # Read raw train data from csv files
            raw_train_data = (
              pipeline
              | '{} - Read Raw Data'.format(step) >> beam.io.textio.ReadFromText(args.raw_train_file)
              | '{} - Remove Empty Rows'.format(step) >> beam.Filter(lambda line: line)
              | '{} - FixCommasAndRemoveFiledTestData'.format(step) >> beam.Map(fix_comma_and_filter_third_column)
              | '{} - Decode CSV Data'.format(step) >> MapAndFilterErrors(converter.decode)

            )

            # create a train dataset from the data and schema
            raw_train_dataset = (raw_train_data, raw_metadata)

            # analyze and transform raw_train_dataset to produced transformed_train_dataset and transform_fn
            transformed_train_dataset, transform_fn = (
                raw_train_dataset
                | '{} - Analyze & Transform'.format(step) >> impl.AnalyzeAndTransformDataset(preprocess)
            )

            # get data and schema separately from the transformed_train_dataset
            transformed_train_data, transformed_metadata = transformed_train_dataset

            # write transformed train data to sink
            _ = (
                transformed_train_data
                | '{} - Write Transformed Data'.format(step) >> beam.io.tfrecordio.WriteToTFRecord(
                    file_path_prefix=args.transform_train_prefix,
                    file_name_suffix=".tfrecords",
                    coder=tft_coders.example_proto_coder.ExampleProtoCoder(transformed_metadata.schema))
            )

            ###### transform eval ##################################################################

            if(runner=='DirectRunner'):
                print("Transform eval data....")

            step = 'eval'

            raw_eval_data = (
              pipeline
              | '{} - Read Raw Data'.format(step) >> beam.io.textio.ReadFromText(args.raw_eval_file)
              | '{} - Remove Empty Lines'.format(step) >> beam.Filter(lambda line: line)
              | '{} - FixCommasAndRemoveFiledTestData'.format(step) >> beam.Map(fix_comma_and_filter_third_column)
              | '{} - Decode CSV Data'.format(step) >> MapAndFilterErrors(converter.decode)

            )

            # create a eval dataset from the data and schema
            raw_eval_dataset = (raw_eval_data, raw_metadata)

            # transform eval data based on produced transform_fn (from analyzing train_data)
            transformed_eval_dataset = (
                (raw_eval_dataset, transform_fn)
                | '{} - Transform'.format(step) >> impl.TransformDataset()
            )

            # get data from the transformed_eval_dataset
            transformed_eval_data, _ = transformed_eval_dataset

            # write transformed eval data to sink
            _ = (
                transformed_eval_data
                | '{} - Write Transformed Data'.format(step) >> beam.io.tfrecordio.WriteToTFRecord(
                    file_path_prefix=args.transform_eval_prefix,
                    file_name_suffix=".tfrecords",
                    coder=tft_coders.example_proto_coder.ExampleProtoCoder(transformed_metadata.schema))
            )


            ###### write transformation metadata #######################################################
            if(runner=='DirectRunner'):
                print("Saving transformation artefacts ....")

            # write transform_fn as tf.graph
            _ = (
                transform_fn
                | 'Write Transform Artefacts' >> transform_fn_io.WriteTransformFn(params.TRANSFORM_ARTIFACTS_DIR)
            )

    if runner=='DataflowRunner':
        pipeline.run()

import shutil
from datetime import datetime
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
      '--raw-train-file',
      help='Raw Train Data File',
      required=True)

parser.add_argument(
      '--raw-eval-file',
      help='Raw Eval Data File',
      required=True)

parser.add_argument(
      '--transform-train-prefix',
      help='transform train file prefix',
      required=True)

parser.add_argument(
      '--transform-eval-prefix',
      help='transform eval file prefix',
      required=True)

parser.add_argument(
      '--runner',
      help='Runner type',
      default='DirectRunner',
      required=True)


args = parser.parse_args()

    
tf.logging.set_verbosity(tf.logging.ERROR)

runner = args.runner # DirectRunner | DataflowRunner

job_name = 'preprocess-data-tft-{}'.format(datetime.utcnow().strftime('%y%m%d-%H%M%S'))
print('Launching {} job {} ... hang on'.format(runner, job_name))
print("")

dataflow_options = {
    'region': 'europe-west1',
    'staging_location': os.path.join(params.DATA_DIR, 'tmp', 'staging'),
    'temp_location': os.path.join(params.DATA_DIR, 'tmp'),
    'job_name': job_name,
    'project': params.GCP_PROJECT_ID,
    'worker_machine_type': 'n1-standard-2',
    'max_num_workers': 20,
    'teardown_policy': 'TEARDOWN_ALWAYS',
    'no_save_main_session': True,
    'requirements_file': 'requirements.txt',
}

if runner == 'DirectRunner':

    shutil.rmtree(params.TRANSFORM_ARTIFACTS_DIR, ignore_errors=True)
    shutil.rmtree(params.TRANSFORMED_DATA_DIR, ignore_errors=True)
    shutil.rmtree(params.TEMP_DIR, ignore_errors=True)


    run_transformation_pipeline(args, dataflow_options)
    print("Transformation is done!")

