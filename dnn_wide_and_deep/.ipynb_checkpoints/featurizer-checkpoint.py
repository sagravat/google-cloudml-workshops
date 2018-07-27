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
from tensorflow.python.feature_column import feature_column
import params
import os
import metadata


def get_vocabulary_file_by_name(transform_artefacts_dir, key):
    return os.path.join(
        transform_artefacts_dir,
        transform_fn_io.TRANSFORM_FN_DIR,
        'assets',
        key.replace('_integerized',''))


def get_vocabulary_size_by_name(transform_artefacts_dir, key):
    vocabulary = get_vocabulary_file_by_name(transform_artefacts_dir, key)
    with tf.gfile.Open(vocabulary, 'r') as f:
        return sum(1 for _ in f)

def extend_feature_columns(feature_columns, hparams):

    """
    add custom features here
    """

    return feature_columns


def create_feature_columns(hparams):
    
    transformed_metadata = metadata_io.read_metadata(
        os.path.join(params.Params.TRANSFORM_ARTIFACTS_DIR,"transformed_metadata"))

    transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

    #print(transformed_feature_spec)

    TRANSFORMED_NUMERIC_FEATURE_NAMES = [
        feature_name
        for  feature_name in transformed_feature_spec.keys()
        if feature_name.endswith('_scaled')
    ]


    TRANSFORMED_BUCKETIZED_FEATURE_NAMES = [
        feature_name
        for  feature_name in transformed_feature_spec.keys()
        if feature_name.endswith('_bucketized')
    ]


    TRANSFORMED_CATEGORICAL_FEATURE_NAMES = metadata.CATEGORICAL_FEATURE_NAMES
    
    TRANSFORMED_INTEGERIZED_CATEGORICAL_FEATURE_NAMES = [
        feature_name
        for  feature_name in transformed_feature_spec.keys()
        if feature_name.endswith('_integerized')
    ]

    print TRANSFORMED_NUMERIC_FEATURE_NAMES
    print ""
    print TRANSFORMED_BUCKETIZED_FEATURE_NAMES
    print ""
    print TRANSFORMED_CATEGORICAL_FEATURE_NAMES
    print ""
    print TRANSFORMED_INTEGERIZED_CATEGORICAL_FEATURE_NAMES

    feature_columns = {}

    numeric_columns = {
        feature_name: tf.feature_column.numeric_column(feature_name)
        for feature_name in TRANSFORMED_NUMERIC_FEATURE_NAMES
    }
    
    bucketized_columns = {
        feature_name: tf.feature_column.categorical_column_with_identity(feature_name, num_buckets=params.Params.NUM_BUCKETS+2)
        for feature_name in TRANSFORMED_BUCKETIZED_FEATURE_NAMES
    }
    
    categorical_columns = {
        feature_name: tf.feature_column.categorical_column_with_vocabulary_file(
            key=feature_name, 
            vocabulary_file=get_vocabulary_file_by_name(params.Params.TRANSFORM_ARTIFACTS_DIR, feature_name))
        for feature_name in TRANSFORMED_CATEGORICAL_FEATURE_NAMES}
    
    integerized_columns = {
        feature_name: tf.feature_column.categorical_column_with_identity(
            key=feature_name, 
            num_buckets=get_vocabulary_size_by_name(params.Params.TRANSFORM_ARTIFACTS_DIR, feature_name))
        for feature_name in TRANSFORMED_INTEGERIZED_CATEGORICAL_FEATURE_NAMES}
    

    if numeric_columns is not None:
        feature_columns.update(numeric_columns)
        
    if bucketized_columns is not None:
        feature_columns.update(bucketized_columns)
        
    if integerized_columns is not None:
        feature_columns.update(integerized_columns)
        
    if categorical_columns is not None:
        feature_columns.update(categorical_columns)

    return extend_feature_columns(feature_columns, hparams)

def get_bucketized_columns(hparams):

    feature_columns = list(create_feature_columns(hparams).values())

    bucketized_columns = list(
        filter(lambda column: 
                isinstance(column, feature_column._BucketizedColumn)
               ,feature_columns)
    )

    return bucketized_columns

def get_wide_deep_columns(hparams):
    
    feature_columns = list(create_feature_columns(hparams).values())
    
    dense_columns = list(
        filter(lambda column: 
                 isinstance(column, feature_column._NumericColumn) 
               | isinstance(column, feature_column._EmbeddingColumn)
               ,feature_columns
        )
    )

    categorical_columns = list(
        filter(lambda column: 
                 isinstance(column, feature_column._VocabularyListCategoricalColumn) 
               | isinstance(column, feature_column._VocabularyFileCategoricalColumn) 
               | isinstance(column, feature_column._IdentityCategoricalColumn) 
               | isinstance(column, feature_column._BucketizedColumn)
               ,feature_columns)
    )
    
    sparse_columns = list(
        filter(lambda column: 
                 isinstance(column,feature_column._HashedCategoricalColumn) 
               | isinstance(column, feature_column._CrossedColumn)
               , feature_columns)
    )

    indicator_columns = []
    
    if hparams.use_indicators: 
        indicator_columns = [
            tf.feature_column.indicator_column(column)
            for column in categorical_columns
        ]
    
    deep_feature_columns = dense_columns + indicator_columns
    wide_feature_columns = (categorical_columns + sparse_columns) if hparams.use_wide_columns else []
    
    return wide_feature_columns, deep_feature_columns


def create_raw_metadata():

    column_schemas = {}

    # ColumnSchema for numeric features
    column_schemas.update({
      key: dataset_schema.ColumnSchema(
          tf.float32, [], dataset_schema.FixedColumnRepresentation())
      for key in metadata.NUMERIC_FEATURE_NAMES
    })

    # ColumnSchema for categorical features
    column_schemas.update({
      key: dataset_schema.ColumnSchema(
          tf.string, [], dataset_schema.FixedColumnRepresentation(default_value="null"))
      for key in metadata.CATEGORICAL_FEATURE_NAMES
    })

    # ColumnSchema for target feature
    column_schemas[metadata.TARGET_FEATURE_NAME] = dataset_schema.ColumnSchema(
        tf.string, [],
        dataset_schema.FixedColumnRepresentation()
    )

    # Dataset Metadata
    raw_metadata = dataset_metadata.DatasetMetadata(
        dataset_schema.Schema(column_schemas)
    )

    return raw_metadata

