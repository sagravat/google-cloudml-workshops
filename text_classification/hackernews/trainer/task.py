# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example implementation of code to run on the Cloud ML service.
"""

import argparse
import json
import os

import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket',
        help = 'GCS path to data. We assume that data is in gs://BUCKET/news/preproc/',
        required = True
    )
    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    parser.add_argument(
        '--train_examples',
        help = 'Number of examples (in thousands) to run the training job over. If this is more than actual # of examples available, it cycles through them. So specifying 1000 here when you have only 100k examples makes this 10 epochs.',
        type = int,
        default = 5000
    )
    parser.add_argument(
        '--batch_size',
        help = 'Number of examples to compute gradient over.',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--nembeds',
        help = 'Embedding size of a cross of n key real-valued parameters',
        type = int,
        default = 3
    )
    parser.add_argument(
        '--nnsize',
        help = 'Hidden layer sizes to use for DNN feature columns -- provide space-separated layers',
        nargs = '+',
        type = int,
        default=[128, 32, 4]
    )
    parser.add_argument(
        '--pattern',
        help = 'Specify a pattern that has to be in input files. For example 00001-of will process only one shard',
        default = 'of'
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )
    parser.add_argument(
        '--eval_steps',
        help = 'Positive number of steps for which to evaluate model. Default to None, which means to evaluate until input_fn raises an end-of-input exception',
        type = int,       
        default = None
    )

    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    TRAIN_SIZE = 73124
    NUM_EPOCHS = 10
    BATCH_SIZE = 1000

    TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS
    EVAL_EVERY_SEC = 60

    hparams  = tf.contrib.training.HParams(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        trainable_embedding=False,
        learning_rate=0.01,
        hidden_units=[256, 128],
        max_steps=TOTAL_STEPS
    )

    output_dir = arguments.pop('output_dir')

    model_dir = os.path.join(output_dir, "hackernews_model")


    run_config = tf.estimator.RunConfig(
        tf_random_seed=19830610,
        log_step_count_steps=1000,
        save_checkpoints_secs=EVAL_EVERY_SEC,
        keep_checkpoint_max=1,
        model_dir=model_dir
    )


    model.BUCKET     = arguments.pop('bucket')
    model.BATCH_SIZE = arguments.pop('batch_size')
    model.TRAIN_STEPS = (arguments.pop('train_examples') * 1000) / model.BATCH_SIZE
    model.EVAL_STEPS = arguments.pop('eval_steps')    
    print ("Will train for {} steps using batch_size={}".format(model.TRAIN_STEPS, model.BATCH_SIZE))
    model.PATTERN = arguments.pop('pattern')
    model.NEMBEDS= arguments.pop('nembeds')
    model.NNSIZE = arguments.pop('nnsize')
    print ("Will use DNN size of {}".format(model.NNSIZE))

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )

    # Run the training job
    model.train_and_evaluate(hparams, run_config)
