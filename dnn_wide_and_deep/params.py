import os

class Params:
    pass

# Set to run on GCP
Params.GCP_PROJECT_ID = ''

Params.TARGET_LABELS = ['0', '1']
Params.NUM_BUCKETS = 4
# change to GCS location to run on GCP
Params.DATA_DIR = 'data'
Params.TRANSFORMED_DATA_DIR = 'data/transformed'

Params.RAW_DATA_DELIMITER = '\t'
Params.RAW_TRAIN_DATA_FILE = os.path.join(Params.DATA_DIR, 'new_train.csv')
Params.RAW_EVAL_DATA_FILE = os.path.join(Params.DATA_DIR, 'new_eval.csv')

Params.TRANSFORMED_TRAIN_DATA_FILE_PREFIX = os.path.join(Params.TRANSFORMED_DATA_DIR, 'my-train')
Params.TRANSFORMED_EVAL_DATA_FILE_PREFIX = os.path.join(Params.TRANSFORMED_DATA_DIR, 'my-eval')

# change to GCS location to run on GCP
Params.TEMP_DIR = 'tmp'

Params.TFMA_OUT =  os.path.join(Params.TEMP_DIR, 'tfma')

# change to GCS location to run on GCP
Params.MODELS_DIR = 'models'

Params.TRANSFORM_ARTIFACTS_DIR = os.path.join(Params.MODELS_DIR,'transform')

Params.TRANSFORM = True

Params.TRAIN = True

Params.EXTEND_FEATURE_COLUMNS = True

Params.RESUME_TRAINING = False

Params.DISPLAY_FACETS = True
