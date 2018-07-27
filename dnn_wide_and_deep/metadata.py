import tensorflow as tf
import os

RAW_FEATURE_NAMES = [
]


NUMERIC_FEATURE_NAMES = [
]

CATEGORICAL_FEATURE_NAMES = [
]

OOV_SIZE = 10
VOCAB_SIZE = 1000

VOCAB_FEATURE_NAMES = [
]

TARGET_FEATURE_NAME = ''

TARGET_LABELS = ['0', '1']

TRAIN_SIZE = 25529
EVAL_SIZE = 4097
NUM_EPOCHS = 50
BATCH_SIZE = 128
TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS
EVAL_EVERY_SEC = 30

hparams  = tf.contrib.training.HParams(
    num_epochs = NUM_EPOCHS,
    batch_size = BATCH_SIZE,

    embedding_size = 3,

    use_indicators = True,
    use_wide_columns = True,
    learning_rate = 0.1,

    hidden_units=[16, 8],
    dropout_prob = 0.0,

    max_steps = TOTAL_STEPS,

)


class Params:
    pass

Params.GCP_PROJECT_ID = ''
Params.NUM_BUCKETS = 4
Params.DATA_DIR = 'data'
Params.TRANSFORMED_DATA_DIR = 'data/transformed'
Params.RAW_DATA_DELIMITER = '\t'
Params.TRANSFORM_ARTIFACTS_DIR = 'models/transform'
Params.TFMA_OUT = 'tmp/tfma'
Params.TEMP_DIR = 'tmp'
Params.MODELS_DIR = 'models'
Params.TRAIN = True
Params.EXTEND_FEATURE_COLUMNS = True
Params.RESUME_TRAINING = False
Params.DISPLAY_FACETS = True
Params.TARGET_LABELS = ['0', '1']

params = {
    'GCP_PROJECT_ID': '',
    'NUM_BUCKETS' : 4,
    'DATA_DIR': 'data', # change to GCS location to run on GCP
    'TRANSFORMED_DATA_DIR': 'data/transformed',
    'RAW_DATA_DELIMITER': '\t',
    #'RAW_TRAIN_DATA_FILE': os.path.join(params.DATA_DIR, 'new_train.csv'),
    #'RAW_EVAL_DATA_FILE': os.path.join(params.DATA_DIR, 'new_eval.csv'),
    #'TRANSFORM_ARTEFACTS_DIR': os.path.join(params.MODELS_DIR,'transform'),
    'TRANSFORM_ARTEFACTS_DIR': 'models/transform',
    #'TFMA_OUT':  os.path.join(params.TEMP_DIR, 'tfma'),
    'TFMA_OUT':  'tmp/tfma',
    'TEMP_DIR': 'tmp', # change to GCS location to run on GCP
    'MODELS_DIR': 'models', # change to GCS location to run on GCP
    'TRANSFORM': True,
    'TRAIN' : True,
    'EXTEND_FEATURE_COLUMNS' : True,
    'RESUME_TRAINING' : False,
    'DISPLAY_FACETS' : True
}
