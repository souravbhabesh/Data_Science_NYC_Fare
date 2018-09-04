import pandas as pd
import numpy as np
import shutil
import tensorflow as tf
print(tf.__version__)
import datetime

tf.logging.set_verbosity(tf.logging.INFO)
CSV_COLUMNS = 'fare_amount,dayofweek,hourofday,pickuplon,pickuplat,dropofflon,dropofflat,passengers,key'.split(',')
LABEL_COLUMN = 'fare_amount'
KEY_FEATURE_COLUMN = 'key'
#Replacing null values with defaults
DEFAULTS = [[0.0], ['Sun'], [0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]
#TRAIN_STEPS = 1000


# These are the raw input columns, and will be provided for prediction also
INPUT_COLUMNS = [
    # Define features
    tf.feature_column.categorical_column_with_vocabulary_list('dayofweek', vocabulary_list = ['Sun', 'Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat']),
    tf.feature_column.categorical_column_with_identity('hourofday', num_buckets = 24),

    # Numeric columns
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
    
    # Engineered features that are created in the input_fn
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean')
]

# Build the estimator
def build_estimator(model_dir, nbuckets, hidden_units):
    """
     Build an estimator starting from INPUT COLUMNS.
     These include feature transformations and synthetic features.
     The model is a wide-and-deep model.
  """

    # Input columns
    (dayofweek, hourofday, plat, plon, dlat, dlon, pcount, latdiff, londiff, euclidean) = INPUT_COLUMNS

    # Bucketize the lats & lons
    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    # Feature cross
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets) #setting up grid
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4 )
    day_hr =  tf.feature_column.crossed_column([dayofweek, hourofday], 24 * 7)

    # Wide columns and deep columns.
    wide_columns = [
        # Feature crosses 
        dloc, ploc, pd_pair,
        day_hr,

        # Sparse columns (#less options)
        dayofweek, hourofday,

        # Anything with a linear relationship
        pcount 
    ]

    deep_columns = [
        # Embedding_column to "group" together ...
        tf.feature_column.embedding_column(pd_pair, 10), #embedding_column in tf goes from pd_pair options to 10 options (controlled PCA)
        tf.feature_column.embedding_column(day_hr, 10),

        # Numeric columns
        plat, plon, dlat, dlon,
        latdiff, londiff, euclidean
    ]

    
    #EVAL_INTERVAL = 30
    #run_config = tf.estimator.RunConfig(save_checkpoints_secs = eval_interval,
                                      #keep_checkpoint_max = 10)
    #Defining the estimator
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = model_dir,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_units = hidden_units)
        #config = run_config)

    # add extra evaluation metric for hyperparameter tuning
    estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)
    return estimator
  
  
### Evaluation metric function
def add_eval_metrics(labels, predictions):
  pred_values = predictions['predictions']
  return {
        'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)
          }


# Create feature engineering function that will be used in the input and serving input functions
def add_engineered(features):
    # this is how you can do feature engineering in TensorFlow
    lat1 = features['pickuplat']
    lat2 = features['dropofflat']
    lon1 = features['pickuplon']
    lon2 = features['dropofflon']
    latdiff = (lat1 - lat2)
    londiff = (lon1 - lon2)
    
    # set features for distance with sign that indicates direction
    features['latdiff'] = latdiff
    features['londiff'] = londiff
    dist = tf.sqrt(latdiff * latdiff + londiff * londiff)
    features['euclidean'] = dist
    return features


# Create serving input function to be able to serve predictions
def serving_input_fn():
    feature_placeholders = {
        # All the real-valued columns
        column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS[2:]
    }
    feature_placeholders['dayofweek'] = tf.placeholder(tf.string, [None])
    feature_placeholders['hourofday'] = tf.placeholder(tf.int32, [None])

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(add_engineered(features), feature_placeholders)


# Create input function to load data into datasets
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return add_engineered(features), label
        
        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
        return batch_features, batch_labels
    return _input_fn


    # Create estimator to train and evaluate
def train_and_evaluate(args):
  #EVAL_INTERVAL = 30
  #run_config = tf.estimator.RunConfig(save_checkpoints_secs = args['eval_interval'],
                                      #keep_checkpoint_max = 3)
  estimator = build_estimator(args['output_dir'], args['nbuckets'], args['hidden_units'])
  '''estimator = tf.estimator.DNNRegressor(
                       model_dir = output_dir,
                       feature_columns = INPUT_COLUMNS,
                       hidden_units = [64, 32],
                       config = run_config)
  '''
  train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(args['train_data_paths'],
                                batch_size = args['train_batch_size'],
                                mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = args['train_steps'])
  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(input_fn = read_dataset(args['eval_data_paths'],
                                batch_size = 10000,
                                mode = tf.estimator.ModeKeys.EVAL),
        steps = None,
        start_delay_secs = args['eval_delay_secs'],
        throttle_secs = args['min_eval_frequency'],
        exporters = exporter)
 # eval_predict = estimator.predict(
 #                      input_fn = read_dataset('taxi-valid.csv', mode = tf.estimator.ModeKeys.PREDICT))
 # print(eval_predict.next())
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# Run the model
#shutil.rmtree('NYC_model', ignore_errors = True) # start fresh each time
#model_nyc = train_and_evaluate(args)