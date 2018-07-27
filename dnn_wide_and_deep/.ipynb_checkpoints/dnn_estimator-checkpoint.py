import tensorflow as tf
import input
import featurizer
import params

def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.Params.TARGET_LABELS)
    )
    return table.lookup(label_string_tensor)


def metric_fn(labels, predictions):

    metrics = {}

    indices = parse_label_column(labels)
    pred_class = predictions['class_ids']
    metrics['micro_accuracy'] = tf.metrics.mean_per_class_accuracy(
        labels=indices,
        predictions=pred_class,
        num_classes=len(params.Params.TARGET_LABELS)
    )

    return metrics

def create_estimator(run_config, hparams):
    
    print "creating a dnn linear combined estimator..."
    print ""
    
    wide_feature_columns, deep_feature_columns = featurizer.get_wide_deep_columns(hparams)
    
    #print "wide columns: {}".format(wide_feature_columns)
    print "wide columns ####################################################"
    for column in wide_feature_columns:
        print(column)
    print ""
    #print "deep columns: {}".format(deep_feature_columns)
    print "deep columns ####################################################"
    for column in deep_feature_columns:
        print(column)

    print ""
    
    linear_optimizer = tf.train.FtrlOptimizer(learning_rate=hparams.learning_rate)
    #dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=hparams.learning_rate)
    dnn_optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate=hparams.learning_rate,
        l2_regularization_strength=1e-4)

    estimator = tf.estimator.DNNLinearCombinedClassifier(
        
        n_classes = len(params.Params.TARGET_LABELS),
        label_vocabulary = params.Params.TARGET_LABELS,
        
        dnn_feature_columns = deep_feature_columns,
        linear_optimizer=linear_optimizer,
        linear_feature_columns=wide_feature_columns,
        
        dnn_hidden_units = hparams.hidden_units,
        
        dnn_optimizer=dnn_optimizer,
        
        dnn_activation_fn = tf.nn.relu,
        dnn_dropout = hparams.dropout_prob,
        
        config= run_config
    )
    
    
    estimator = tf.contrib.estimator.add_metrics(
        estimator=estimator,
        metric_fn=metric_fn
    )
    
    return estimator



