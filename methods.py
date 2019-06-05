"""
Sesarch session segmentation using a number of methods
"""

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
from keras.optimizers import SGD
from keras.initializers import VarianceScaling

import datasets, time, DEC, IDEC, IDEC_DEC, SymDEC, metrics
import os, gc, time, shutil, glob

import numpy as np
import tensorflow as tf

# Random seed and TF logging
tf.logging.set_verbosity(tf.logging.INFO)
np.random.seed(43)

def unsupervised_accuracy(true_labels, predicted_labels):
    """
    Calculate unsupervised clustering accuracy using the Hungarian algorithm
    in scikit-learn

    [1] https://github.com/XifengGuo/IDEC
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)

    n_labels = predicted_labels.size
    n_clusters = max(predicted_labels.max(), true_labels.max()) + 1
    weights = np.zeros((n_clusters, n_clusters), dtype=np.int64)

    for i in range(n_labels):
        weights[predicted_labels[i], true_labels[i]] += 1
    
    indices = linear_assignment(weights.max() - weights)
    accuracy = float(sum([weights[i, j] for i, j in indices])) / n_labels
    return accuracy

def pairwise_counts(true_labels, predicted_labels):
  f00, f01, f10, f11 = 0.0, 0.0, 0.0, 0.0

  assert(len(true_labels) == len(predicted_labels))

  for i in range(0, len(true_labels) - 1):
    for j in range(i + 1, len(true_labels)):
      if true_labels[i] == true_labels[j] and predicted_labels[i] == predicted_labels[j]:
        f11 += 1.0
      elif true_labels[i] != true_labels[j] and predicted_labels[i] != predicted_labels[j]:
        f00 += 1.0
      elif true_labels[i] == true_labels[j] and predicted_labels[i] != predicted_labels[j]:
        f01 += 1.0
      elif true_labels[i] != true_labels[j] and predicted_labels[i] == predicted_labels[j]:
        f10 += 1.0
    
  return f00, f01, f10, f11

def rand_index(true_labels, predicted_labels):
  f00, f01, f10, f11 = pairwise_counts(true_labels, predicted_labels)
  return (f00 + f11) / (f00 + f01 + f10 + f11 + 1e-10)

def jaccard_index(true_labels, predicted_labels):
  _, f01, f10, f11 = pairwise_counts(true_labels, predicted_labels)
  return f11 / (f01 + f10 + f11 + 1e-10)
  
def kmeans(data, labels):
  """
  Cluster data by running kmeans implementation in scikit-learn

  [1] https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
  """

  n_clusters = np.amax(labels)
  if np.amin(labels) == 0:
    n_clusters += 1
  print("Number of clusters: " + str(n_clusters))
  
  km = KMeans(n_clusters=n_clusters)

  return km.fit_predict(data, labels)

def time_partition(dataset):
  """
  Consider one task per time session in AOL (Lucchese et al., 2011) dataset. 
  Time IDs are based on a time threshold of 26 minutes
  """
  
  fscores = []
  predicted_labels = []
  true_labels = []
  pairwise_predicted = []
  pairwise_true = []

  for i in range(len(dataset.time_ids) - 1):
    true_labels.append(dataset.labels[i])
    predicted_labels.append(1)
    
    if dataset.time_ids[i] != dataset.time_ids[i + 1]:
      # Time session has ended
      true_labels = true_labels
      predicted_labels = predicted_labels

      fscores.append(
        metrics.f1_score(true_labels, predicted_labels, average='macro'))

      if len(predicted_labels) > 5:
        pairwise_predicted.extend(predicted_labels)
        pairwise_true.extend(true_labels)

      true_labels = []
      predicted_labels = []

  rand = rand_index(pairwise_true, pairwise_predicted)
  jaccard = jaccard_index(pairwise_true, pairwise_predicted)

  print('F-measure ' + str(np.average(fscores)))
  print('Rand Index ' + str(rand))
  print('Jaccard Index ' + str(jaccard))

def time_segmentation(dataset):
  """
  Two queries are from different tasks if they pertain to a different time 
  session. Time IDs are based on a time threshold of 26 minutes
  """

  predicted_labels = []
  true_labels = []

  for i in range(len(dataset.labels) - 1):
    true_labels.append(dataset.labels[i])

    if dataset.same_tids[i] == 0:
      predicted_labels.append(1)
    else:
      predicted_labels.append(0)

  print("Accuracy: %0.3f" % metrics.accuracy_score(
    true_labels, predicted_labels))
  print("F-score: %0.3f" % metrics.f1_score(
    true_labels, predicted_labels))

def symdec(dataset, save_dir):
  """
  Run SymDEC

  """

  x = dataset.data
  y = dataset.labels

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  ae_weights = save_dir + '/ae_weights.h5'
  n_clusters = np.max(y) + 1
  batch_size = 256
  init = VarianceScaling(scale=1. / 3., mode='fan_in',
    distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)

  # Pretrain autoencoder
  if not os.path.exists(ae_weights):
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    pretrain_epochs = 50
    dec = DEC.DEC(
      dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)
    dec.pretrain(x=x, y=y, optimizer=pretrain_optimizer, epochs=pretrain_epochs,
      batch_size=batch_size, save_dir=save_dir)

  # Run DEC
  optimizer = SGD(lr=0.1, momentum=0.99)
  gamma = 0.1
  maxiter = 1200
  update_interval = 50 
  tol = 0.001
  
  # prepare the DEC model
  dec = SymDEC.DEC(
    dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)
  dec.autoencoder.load_weights(ae_weights)
  dec.model.summary()

  # begin clustering, time not include pretraining part.
  t0 = time.time()
  dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')
  y_pred = dec.fit(x, y=y, tol=tol, maxiter=maxiter, batch_size=batch_size, 
    update_interval=update_interval, save_dir=save_dir)

  accuracy = IDEC_DEC.cluster_acc(y, y_pred)
  print ('acc:', accuracy)
  print ('clustering time: ', (time.time() - t0))

  return accuracy, metrics.nmi(y, y_pred), metrics.ari(y, y_pred)

def dec(dataset, save_dir):
  """
  Run DEC

  [1] https://github.com/XifengGuo/IDEC
  """

  x = dataset.data
  y = dataset.labels

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  ae_weights = save_dir + '/ae_weights.h5'
  n_clusters = np.max(y) + 1
  batch_size = 256
  init = VarianceScaling(scale=1. / 3., mode='fan_in',
    distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
    
  # Pretrain autoencoder
  if not os.path.exists(ae_weights):
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    pretrain_epochs = 50
    dec = DEC.DEC(
      dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)
    dec.pretrain(x=x, y=y, optimizer=pretrain_optimizer, epochs=pretrain_epochs,
      batch_size=batch_size, save_dir=save_dir)

  # Run DEC
  optimizer = SGD(lr=0.1, momentum=0.99)
  gamma = 0.1
  maxiter = 2e4
  update_interval = 50 
  tol = 0.001
  
  # prepare the DEC model
  dec = DEC.DEC(
    dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)
  dec.autoencoder.load_weights(ae_weights)
  dec.model.summary()

  # begin clustering, time not include pretraining part.
  t0 = time.time()
  dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')
  y_pred = dec.fit(x, y=y, tol=tol, maxiter=maxiter, batch_size=batch_size, 
    update_interval=update_interval, save_dir=save_dir)

  accuracy = IDEC_DEC.cluster_acc(y, y_pred)
  print ('acc:', accuracy)
  print ('clustering time: ', (time.time() - t0))

  return accuracy, metrics.nmi(y, y_pred), metrics.ari(y, y_pred)

def idec(dataset, save_dir):
  """
  Run IDEC

  [1] https://github.com/XifengGuo/IDEC
  """

  x = dataset.data
  y = dataset.labels

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  ae_weights = save_dir + '/ae_weights.h5'
  n_clusters = np.max(y) + 1
  batch_size = 256
  init = VarianceScaling(scale=1. / 3., mode='fan_in',
    distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
    
  # Pretrain autoencoder
  if not os.path.exists(ae_weights):
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    pretrain_epochs = 30
    dec = DEC.DEC(
      dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)
    dec.pretrain(x=x, y=y, optimizer=pretrain_optimizer, epochs=pretrain_epochs,
      batch_size=batch_size, save_dir=save_dir)

  # Run IDEC
  optimizer = SGD(lr=0.1, momentum=0.99)
  gamma = 0.1
  maxiter = 2e4
  update_interval = 50 
  tol = 0.005
  
  # prepare the IDEC model
  idec = IDEC.IDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, batch_size=batch_size)
  idec.initialize_model(ae_weights=ae_weights, gamma=gamma, optimizer=optimizer)
  idec.model.summary()

  # begin clustering, time not include pretraining part.
  t0 = time.time()
  y_pred = idec.clustering(
    x, y=y, tol=tol, maxiter=maxiter, update_interval=update_interval, 
    save_dir=save_dir)
  
  accuracy = IDEC_DEC.cluster_acc(y, y_pred)
  print ('acc:', accuracy)
  print ('clustering time: ', (time.time() - t0))

  return accuracy, metrics.nmi(y, y_pred), metrics.ari(y, y_pred)

class brnn(object):
  """
  Define a bidirectional RNN usign GRU cells and attention mechanism

  [1] https://www.tensorflow.org/tutorials/
  [2] https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
  """
  def __init__(self):
    self.MODEL_DIR     = 'models/rnn_model'
    self.BATCH_SIZE    = 128
    self.LEARNING_RATE = 1e-3
    self.ITERATIONS    = 10000    
    self.HIDDEN_UNITS  = 32
    self.NUM_CLASSES   = 2

  def bidirectional_rnn(self, input_layer):
    """
    Bidirectional RNN in tensorflow
    """
    
    forward_cell = tf.contrib.rnn.GRUCell(
      num_units=self.HIDDEN_UNITS,
      activation=tf.nn.tanh,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      )
    backward_cell = tf.contrib.rnn.GRUCell(
      num_units=self.HIDDEN_UNITS,
      activation=tf.nn.tanh,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      )

    rnn_outputs, output_state_fw, output_state_bw = \
      tf.nn.static_bidirectional_rnn(
      cell_fw=forward_cell,
      cell_bw=backward_cell,
      inputs=input_layer,
      initial_state_fw=None,
      initial_state_bw=None,
      dtype=tf.float32,
      sequence_length=None,
      scope=None
      )

    # Attention mechanism
    last_output = rnn_outputs[-1]
    attention_num_units = last_output.shape[-1]
    attention_state = tf.concat([output_state_fw, output_state_bw], axis=1)
    attention_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
    
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
      num_units=attention_num_units, memory=attention_outputs)
    alignments, _ = attention_mechanism(
      query=last_output, state=attention_state)
    expanded_alignments = tf.expand_dims(alignments, 1)
    context = tf.matmul(expanded_alignments, attention_mechanism.values)
    context = tf.squeeze(context, [1])

    output_attention = tf.concat(
      [context, output_state_fw, output_state_bw], 1)

    output_state = tf.layers.dense(
      inputs=output_attention, 
      units=4*self.HIDDEN_UNITS, 
      activation=tf.nn.tanh,
      use_bias=True, 
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      )

    return output_state, alignments

  def rnn_model_fn(self, features, labels, mode):
    """
    Instantiate a RNN and setup the model for training and inference
    """
    input_layer = tf.unstack(features["data"], axis=2)
    print(np.asarray(input_layer).shape)
    output_state, alignments = self.bidirectional_rnn(input_layer)

    # Classification layer
    output_state = tf.layers.dropout(
      inputs=output_state, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)
    output = tf.layers.dense(
      inputs=output_state, 
      units=self.NUM_CLASSES, 
      activation=None,
      use_bias=True,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      )

    # Generate predictions
    predictions = {
      "labels": tf.argmax(input=output, axis=1),
      "alignments": alignments,
      "probabilities": tf.contrib.layers.softmax(output)
      }

    # Prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Loss function (TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=labels, depth=self.NUM_CLASSES)
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=output)

    # Training operation
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer( 
        learning_rate=self.LEARNING_RATE,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08
        )
      train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Evaluation metrics
    accuracy  = tf.metrics.accuracy(
      labels=labels, predictions=predictions["labels"])
    precision  = tf.metrics.precision(
      labels=labels, predictions=predictions["labels"])
    recall  = tf.metrics.recall(
      labels=labels, predictions=predictions["labels"])
    eval_metric_ops = {
      "accuracy"    : accuracy,
      "precision"   : precision,
      "recall"      : recall
    }
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  def train_test(self, dataset):
    """
    Run training procedure and test the model using dataset
    """
    train_data, test_data, train_labels, test_labels = dataset.split_dataset()
    length  = len(train_data[0])
    train_data = train_data.reshape([-1, 1, length])
    test_data  = test_data .reshape([-1, 1, length])
    
    # Clean model directory
    if os.path.isfile(self.MODEL_DIR + '/checkpoint'):
      os.remove(self.MODEL_DIR + '/checkpoint')
    for model_file in glob.glob('/'.join([self.MODEL_DIR, 'model.ckpt*'])):
      os.remove(model_file)
    
    # Configure GPU memory usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Estimator
    classifier = tf.estimator.Estimator(
      model_fn=self.rnn_model_fn, 
      model_dir=self.MODEL_DIR, 
      config=tf.contrib.learn.RunConfig(
        tf_random_seed=43, session_config=config))
    
    # Train the model computing  metrics over the test set
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"data": train_data},
      y=train_labels,
      batch_size=self.BATCH_SIZE,
      num_epochs=None, # Continue until training steps are finished
      shuffle=True
      )
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"data": test_data},
      y=test_labels,
      batch_size=self.BATCH_SIZE,
      num_epochs=1, 
      shuffle=False
      )
    experiment = tf.contrib.learn.Experiment(
      estimator=classifier,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=self.ITERATIONS,
      eval_steps=None, # evaluate runs until input is exhausted
      eval_delay_secs=120, 
      train_steps_per_iteration=1000
      )
    experiment.continuous_train_and_eval()  

if __name__ == "__main__":
  print('Methods module')
