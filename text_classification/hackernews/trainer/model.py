#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import data

tf.logging.set_verbosity(tf.logging.INFO)


BUCKET = "agravat-demo"
PROJECT = "agravat-demo"
REGION = "us-central1"

os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION

class Params:
    pass


Params.PLATFORM = 'GCP' # local | GCP

Params.DATA_DIR = 'data/news'  if Params.PLATFORM == 'local' else 'gs://{}/data/news'.format(BUCKET)
Params.TRANSFORMED_DATA_DIR = os.path.join(Params.DATA_DIR, 'transformed')

Params.RAW_TRAIN_DATA_FILE_PREFEX = os.path.join(Params.DATA_DIR, 'train')
Params.RAW_EVAL_DATA_FILE_PREFEX = os.path.join(Params.DATA_DIR, 'eval')

Params.MODELS_DIR = 'models/news' if Params.PLATFORM == 'local' else 'gs://{}/models/news'.format(BUCKET)

Params.TEMP_DIR = os.path.join(Params.DATA_DIR, 'tmp')

Params.TRANSFORM = True

Params.TRAIN = True

Params.RESUME_TRAINING = False

Params.EAGER = False

if Params.EAGER:
    tf.enable_eager_execution()

RAW_HEADER = 'key,title,source'.split(',')
RAW_DEFAULTS = [['NA'],['NA'],['NA']]
TARGET_FEATRUE_NAME = 'source'
TARGET_LABELS = ['arstechnica', 'blogspot', 'github', 'medium', 'nytimes', 'techcrunch', 'wired', 'wsj']

TEXT_FEATURE_NAME = 'title'
KEY_COLUMN = 'key'

TRAIN_SIZE = 73124
NUM_EPOCHS = 10
BATCH_SIZE = 1000

TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS
EVAL_EVERY_SEC = 60

def parse_tsv(tsv_row):

    columns = tf.decode_csv(tsv_row, record_defaults=RAW_DEFAULTS, field_delim='\t')
    features = dict(zip(RAW_HEADER, columns))

    features.pop(KEY_COLUMN)
    target = features.pop(TARGET_FEATRUE_NAME)

    return features, target


def generate_tsv_input_fn(files_pattern,
                          mode=tf.estimator.ModeKeys.EVAL,
                          num_epochs=1,
                          batch_size=200):


    def _input_fn():

        #file_names = data.Dataset.list_files(files_pattern)
        file_names = tf.matching_files(files_pattern)

        if Params.EAGER:
            print(file_names)

        dataset = data.TextLineDataset(file_names)

        dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(count=num_epochs,
                                                   buffer_size=batch_size*2)
        )

        dataset = dataset.apply(
                tf.contrib.data.map_and_batch(parse_tsv,
                                              batch_size=batch_size,
                                              num_parallel_batches=2)
        )

        datset = dataset.prefetch(batch_size)

        if Params.EAGER:
            return dataset

        iterator = dataset.make_one_shot_iterator()
        features, target = iterator.get_next()
        return features, target

    return _input_fn


def create_feature_columns(hparams):

    title_embeding_column = hub.text_embedding_column(
        "title", "https://tfhub.dev/google/universal-sentence-encoder/1")

    feature_columns = [title_embeding_column]

    print("feature columns: \n {}".format(feature_columns))
    print("")

    return feature_columns



def create_estimator_hub(hparams, run_config):

    feature_columns = create_feature_columns(hparams)

    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes =len(TARGET_LABELS),
        label_vocabulary=TARGET_LABELS,
        hidden_units=hparams.hidden_units,
        optimizer=optimizer,
        config=run_config
    )


    return estimator



def generate_serving_input_fn():

    def _serving_fn():

        receiver_tensor = {
          'title': tf.placeholder(dtype=tf.string, shape=[None])
        }

        return tf.estimator.export.ServingInputReceiver(
            receiver_tensor, receiver_tensor)

    return _serving_fn



def train_and_evaluate(hparams, run_config):
  train_spec = tf.estimator.TrainSpec(
    input_fn = generate_tsv_input_fn(
        Params.RAW_TRAIN_DATA_FILE_PREFEX+"*",
        mode = tf.estimator.ModeKeys.TRAIN,
        num_epochs=hparams.num_epochs,
        batch_size=hparams.batch_size
    ),
    max_steps=hparams.max_steps,
    hooks=None
  )

  eval_spec = tf.estimator.EvalSpec(
    input_fn = generate_tsv_input_fn(
        Params.RAW_EVAL_DATA_FILE_PREFEX+"*",
        mode=tf.estimator.ModeKeys.EVAL,
        num_epochs=1,
        batch_size=hparams.batch_size
    ),
    exporters=[tf.estimator.LatestExporter(
        name="estimate", # the name of the folder in which the model will be exported to under export
        serving_input_receiver_fn=generate_serving_input_fn(),
        exports_to_keep=1,
        as_text=False)],
    steps=None,
    throttle_secs=EVAL_EVERY_SEC
  )


  estimator = create_estimator_hub(hparams, run_config)

  tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
  )

