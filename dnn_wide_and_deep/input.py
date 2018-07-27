import tensorflow as tf
from tensorflow import data
from tensorflow_transform.tf_metadata import metadata_io
import params
import os
import metadata

def parse_tf_example(tf_example):

    transformed_metadata = metadata_io.read_metadata(os.path.join(params.Params.TRANSFORM_ARTIFACTS_DIR,"transformed_metadata"))
    transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

    parsed_features = tf.parse_example(serialized=tf_example, features=transformed_feature_spec)
    target = parsed_features.pop(metadata.TARGET_FEATURE_NAME)

    return parsed_features, target

# to be applied in traing and serving
# ideally, you put this logic in preprocess_tft, to avoid transforming the records during training several times

def process_features(features):
    return features

def generate_tfrecords_input_fn(files_name_pattern,
                                mode=tf.estimator.ModeKeys.EVAL,
                                num_epochs=1,
                                batch_size=500):

    def _input_fn():

        shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

        file_names = data.Dataset.list_files(files_name_pattern)

        dataset = data.TFRecordDataset(filenames=file_names)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)

        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example))
        dataset = dataset.map(lambda features, target: (process_features(features), target))
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()

        features, target = iterator.get_next()
        return features, target

    return _input_fn
