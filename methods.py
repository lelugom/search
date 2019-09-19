"""
Sesarch session segmentation using a number of methods
"""

import datasets, time, DEC, IDEC, SymDEC

import numpy as np
import networkx as nx
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.utils.linear_assignment_ import linear_assignment
from keras.optimizers import SGD
from keras.initializers import VarianceScaling

from sklearn.cluster import k_means_
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

import os, gc, time, shutil, glob
from distutils.dir_util import copy_tree

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
    n_clusters = max(
      get_num_clusters(true_labels), get_num_clusters(predicted_labels)) + 1
    weights = np.zeros((n_clusters, n_clusters), dtype=np.int64)

    for i in range(n_labels):
      weights[predicted_labels[i], true_labels[i]] += 1
    
    indices = linear_assignment(weights.max() - weights)
    accuracy = float(sum([weights[i, j] for i, j in indices])) / n_labels
    return accuracy

def pairwise_counts(true_labels, predicted_labels):
  tn, fn, fp, tp = 0.0, 0.0, 0.0, 0.0

  assert(len(true_labels) == len(predicted_labels))

  for i in range(0, len(true_labels) - 1):
    for j in range(i + 1, len(true_labels)):
      if true_labels[i] == true_labels[j] and predicted_labels[i] == predicted_labels[j]:
        tp += 1.0 
      elif true_labels[i] != true_labels[j] and predicted_labels[i] != predicted_labels[j]:
        tn += 1.0 
      elif true_labels[i] == true_labels[j] and predicted_labels[i] != predicted_labels[j]:
        fn += 1.0 
      elif true_labels[i] != true_labels[j] and predicted_labels[i] == predicted_labels[j]:
        fp += 1.0 
    
  return tn, fn, fp, tp

def rand_index(true_labels, predicted_labels):
  tn, fn, fp, tp = pairwise_counts(true_labels, predicted_labels)
  return (tn + tp) / (tn + fn + fp + tp + 1e-10)

def jaccard_index(true_labels, predicted_labels):
  _, fn, fp, tp = pairwise_counts(true_labels, predicted_labels)
  return tp / (fn + fp + tp + 1e-10)

def pairwise_fscore(true_labels, predicted_labels):
  _, fn, fp, tp = pairwise_counts(true_labels, predicted_labels)

  precision = tp / (tp + fp + 1e-10)
  recall = tp / (tp + fn + 1e-10)
  pairwise_fscore = 2 * precision * recall / (precision + recall + 1e-10)

  return pairwise_fscore, precision, recall

def cs_recall(true_labels, predicted_labels, denominator=36768.0):
  _, _, _, tp = pairwise_counts(true_labels, predicted_labels)
  return tp / (denominator + 1e-10)

def print_metrics(true_labels, predicted_labels):
  """ 
  Command line output for metrics
  """

  print("Metrics: ")
  print("\tAccuracy: %0.5f" % unsupervised_accuracy(
    true_labels, predicted_labels))
  print("\tNormalized Mutal Info: %.5f" % metrics.normalized_mutual_info_score(
    true_labels, predicted_labels))
  print("\tAdjusted Rand Index: %.5f" % metrics.adjusted_rand_score(
    true_labels, predicted_labels))
  print("\tRand Index: %.5f" % rand_index(
    true_labels, predicted_labels))
  print("\tJaccard Index: %.5f" % jaccard_index(
    true_labels, predicted_labels))
  print("\tHomogeneity: %0.5f" % metrics.homogeneity_score(
    true_labels, predicted_labels))
  print("\tCompleteness: %0.5f" % metrics.completeness_score(
    true_labels, predicted_labels))
  print("\tV-measure: %0.5f" % metrics.v_measure_score(
    true_labels, predicted_labels))
  print("\tFscore: %0.5f" % metrics.f1_score(
    true_labels, predicted_labels, average='micro'))
  print("\tPrecision: %0.5f" % metrics.precision_score(
    true_labels, predicted_labels, average='micro'))
  print("\tRecall: %0.5f" % metrics.recall_score(
    true_labels, predicted_labels, average='micro'))

  fscore, p, r = pairwise_fscore(true_labels, predicted_labels)
  cs_r = cs_recall(true_labels, predicted_labels)
  print("\tPairwise Fscore: %0.5f" % fscore)
  print("\tPairwise Precision: %0.5f" % p)
  print("\tPairwise Recall: %0.5f" % r)
  print("\tPairwise CS Recall: %0.5f \n" % cs_r)  

def get_num_clusters(labels):
  """ Compute the number of clusters from the dataset labels """
  num_clusters = np.max(labels)
  if np.min(labels) == 0:
    num_clusters += 1
  return num_clusters

def kmeans(data, labels):
  """
  Cluster data by running kmeans implementation in scikit-learn

  [1] https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
  """

  n_clusters = get_num_clusters(labels)
  print("KMeans. Number of clusters: " + str(n_clusters))
  
  km = KMeans(n_clusters=n_clusters)
  predicted_labels = km.fit_predict(data, labels)

  print_metrics(labels, predicted_labels)
  return predicted_labels

def kmeans_cosine(data, labels):
  """
  Cluster data by running kmeans implementation in scikit-learn, but using
  cosine similarity instead of euclidean distance 
  
  [1] https://gist.github.com/jaganadhg/b3f6af86ad99bf6e9bb7be21e5baa1b5
  """

  n_clusters = get_num_clusters(labels)
  print("KMeans cosine. Number of clusters: " + str(n_clusters))

  def cosine_dist(X, Y = None, Y_norm_squared = None, squared = False):
    return cosine(X, Y)

  k_means_.euclidean_distances = cosine_dist
  scaler = StandardScaler(with_mean=False)
  sparse_data = scaler.fit_transform(data)
  kmeans = k_means_.KMeans(n_clusters=n_clusters, n_jobs=20, random_state = 43)
  kmeans.fit(sparse_data)
  predicted_labels = kmeans.labels_
  
  print_metrics(labels, predicted_labels)
  return predicted_labels

def dbscan(data, labels):
  """
  Cluster data using DBScan 

  [1] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
  """

  print("DBScan ")
  dbscan = DBSCAN(eps=0.5, min_samples=5)  
  predicted_labels = dbscan.fit_predict(data)

  print_metrics(labels, predicted_labels)
  return predicted_labels

def aglomerative_clustering(data, labels):
  """
  Cluster data using aglomerative implementation in Scikit learn, ward linkage

  [1] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
  """
  
  n_clusters = get_num_clusters(labels)
  print("AgglomerativeClustering. Number of clusters: " + str(n_clusters))

  aglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
  predicted_labels = aglomerative.fit_predict(data)
  
  print_metrics(labels, predicted_labels)
  return predicted_labels

def time_partition(dataset):
  """
  Consider one task per time session in AOL datasets. 
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

def decs(
  dataset, 
  save_dir, 
  runs=10,
  pretrain_epochs=30,
  batch_size=256,
  gamma=0.1,
  maxiter=2e4,
  update_interval=30,
  tol=0.001
  ):
  """
  Run SymDEC, DEC, and IDEC for the dataset
  """

  dirs = [
    'models/symdec/'+save_dir,'models/dec/'+save_dir,'models/idec/'+save_dir]
  methods = [SymDEC.DEC, DEC.DEC, IDEC.IDEC]

  for k in range(len(dirs)):
    save_dir = dirs[k]
    method = methods[k]
    accs, nmis, aris, fscs = [], [], [], []
    for _ in range(runs):
      acc, nmi, ari, fsc = dec(
        dataset, 
        save_dir,
        method,
        pretrain_epochs,
        batch_size,
        gamma,
        maxiter,
        update_interval,
        tol
        )
      accs.append(acc); nmis.append(nmi); aris.append(ari); fscs.append(fsc)
    print("\n\nTest results\n")
    print('\tacc : mean = %.5f  stdev = %.5f' % (np.mean(accs), np.std(accs))) 
    print('\tnmi : mean = %.5f  stdev = %.5f' % (np.mean(nmis), np.std(nmis))) 
    print('\tari : mean = %.5f  stdev = %.5f' % (np.mean(aris), np.std(aris))) 
    print('\tfsc : mean = %.5f  stdev = %.5f' % (np.mean(fscs), np.std(fscs))) 
    
    print(accs); print(nmis); print(aris);  print(fscs)

def dec(
  dataset, 
  save_dir, 
  method=SymDEC.DEC,
  pretrain_epochs=30,
  batch_size=256,
  gamma=0.1,
  maxiter=2e4,
  update_interval=50,
  tol=0.001
  ):
  """
  Run deep embedding clustering method on the provided dataset. Pretrain 
  autoencoder if there are no autoencoder weights in the resutls directory
  """

  x = dataset.data
  y = dataset.labels

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  ae_weights = save_dir + '/ae_weights.h5'
  n_clusters = get_num_clusters(y)
  
  init = VarianceScaling(scale=1. / 3., mode='fan_in',
    distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)

  # Pretrain autoencoder
  if not os.path.exists(ae_weights):
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    
    dec = DEC.DEC(
      dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)
    dec.pretrain(x=x, y=y, optimizer=pretrain_optimizer, epochs=pretrain_epochs,
      batch_size=batch_size, save_dir=save_dir)

  # Prepare model and perform clustering
  optimizer = SGD(lr=0.1, momentum=0.99)
  if method == IDEC.IDEC:
    dec = method(dims=[x.shape[-1], 500, 500, 2000, 10], 
      n_clusters=n_clusters, batch_size=batch_size)
    dec.initialize_model(
      ae_weights=ae_weights, gamma=gamma, optimizer=optimizer)
    dec.model.summary()

    t0 = time.time()
    y_pred = dec.clustering(x, y=y, tol=tol, maxiter=maxiter,
      update_interval=update_interval, save_dir=save_dir)
  else:
    dec = method(dims=[x.shape[-1], 500, 500, 2000, 10], 
      n_clusters=n_clusters, init=init)
    dec.autoencoder.load_weights(ae_weights)
    dec.model.summary()

    t0 = time.time()
    dec.compile(optimizer=optimizer, loss='kld')
    y_pred = dec.fit(x, y=y, tol=tol, maxiter=maxiter, batch_size=batch_size, 
      update_interval=update_interval, save_dir=save_dir)

  acc = unsupervised_accuracy(y, y_pred)
  nmi = metrics.normalized_mutual_info_score(y, y_pred)
  ari = metrics.adjusted_rand_score(y, y_pred)
  fscore, _, _ = pairwise_fscore(y, y_pred)

  print ('\nacc: ', acc)
  print ('clustering time: ', (time.time() - t0))

  return acc, nmi, ari, fscore

class qcwcc(object):
  """
  Clustering method based on graphs. It gives the best results for task session
  identification in (Lucchese et al., 2011)
  """
  def __init__(self, data, labels, threshold=0.7, alpha=0.5, 
    representation=datasets.representation().word2vec, 
    semantic=datasets.clueweb().semantic_similarity):
    self.data = data
    self.labels = labels
    self.threshold = threshold
    self.alpha = alpha
    self.graph = nx.Graph()
    self.predicted_labels = None

    self.compute_representation = representation
    self.compute_semantic = semantic

  def query_similarity(self, q0, q1):
    """
    Compute similarity between the word embeddings representing queries. Use 
    cosine distance for lexical relatedness and cluweb for semantic relatedness
    """
    lexical = self.compute_representation([q0, q1])
    lexical_sim = 1 - cosine(lexical[0], lexical[1])
    semantic_sim = self.compute_semantic(q0, q1)
    sim = self.alpha * lexical_sim + (1 - self.alpha) * semantic_sim
    return sim

  def build_graph(self):
    """
    Use query indexes as nodes and query distances as weights
    """
    assert isinstance(self.data[0], str)

    for i in range(len(self.data) - 1):
      for j in range(i+1, len(self.data)):
        weight = self.query_similarity(self.data[i], self.data[j])
        self.graph.add_edge(i, j, weight=weight)
      print('\tbuilding graph %d'%(i))

  def prune_graph(self):
    """
    Use threshold to discard edges with queries too distant apart
    """
    weak_edges = []
    for u, v, weight in self.graph.edges.data('weight'): 
      if weight < self.threshold or np.isnan(weight):
        weak_edges.append((u, v)) 
    self.graph.remove_edges_from(weak_edges)

  def label_queries(self):
    """
    Use connected components in the pruned graph to detect clusters and label
    queries accordingly
    """
    cluster = 0
    predicted_labels = np.zeros(self.labels.shape)
    for component in nx.connected_components(self.graph):
      for idx in component:
        predicted_labels[idx] = cluster
      cluster += 1
    self.predicted_labels = np.asarray(predicted_labels, dtype=np.int32)

  def cluster(self):
    """
    Run QC_WCC graph based method for queries clustering
    """
    self.build_graph()
    self.prune_graph()
    self.label_queries()

    print('QC WCC threshold %.2f alpha %.2f'%(self.threshold, self.alpha))
    print_metrics(self.labels, self.predicted_labels)
    return self.predicted_labels

def qcwcc_time_session(dataset, threshold=0.7, alpha=0.5):
  """
  Perform clustering using the 26 minutes time gap session in Lucchese et al., 
  2011. Use a dataset with both Lucchese et al., 2011 labels and Sen et al., 
  2018 cross session task labels
  """

  data, labels = [], []
  fscores, weights, precisions, recalls = [], [], [], []
  
  for i in range(len(dataset.data) - 1):
    data.append(dataset.data[i])
    labels.append(dataset.cross_session_labels[i])

    if dataset.time_ids[i+1] != dataset.time_ids[i]:
      w = len(labels)
      data = np.asarray(data, dtype=np.float32)
      labels = np.asarray(labels, dtype=np.int32)
      predicted_labels = qcwcc(
        data, labels, threshold=threshold).cluster()

      if w > 1:
        fs, p, r = pairwise_fscore(labels, predicted_labels)
        weights.append(w)
        fscores.append(fs)
        precisions.append(p)
        recalls.append(r)

      data, labels = [], []
        
  fscore = np.average(fscores, weights=weights)
  precision = np.average(precisions, weights=weights)
  recall = np.average(recalls, weights=weights)

  print("\tSession Fscore: %0.5f" % fscore)
  print("\tSession Precision: %0.5f" % precision)
  print("\tSession Recall: %0.5f \n\n" % recall)

def gelu(x):
  """
  Activation function from https://github.com/google-research/bert

  Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

class brnn(object):
  """
  Define a bidirectional RNN usign GRU cells and attention mechanism

  [1] https://www.tensorflow.org/tutorials/
  [2] https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
  """
  def __init__(self):
    self.MODEL_DIR     = 'models/rnn_model'
    self.BATCH_SIZE    = 128
    self.LEARNING_RATE = 1e-4
    self.ITERATIONS    = 60000    
    self.HIDDEN_UNITS  = 32
    self.NUM_CLASSES   = 2
    self.CELL          = 'GRU'
    self.ACTIVATION    = tf.nn.tanh

  def bidirectional_rnn(self, input_layer):
    """
    Bidirectional RNN in tensorflow
    """
    if self.CELL == 'GRU':
      forward_cell = tf.contrib.rnn.GRUCell(
        num_units=self.HIDDEN_UNITS,
        activation=self.ACTIVATION,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        )
      backward_cell = tf.contrib.rnn.GRUCell(
        num_units=self.HIDDEN_UNITS,
        activation=self.ACTIVATION,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        )
    elif self.CELL == 'LSTM':
      forward_cell = tf.contrib.rnn.LSTMCell(
        num_units=self.HIDDEN_UNITS,
        use_peepholes = True,
        initializer=tf.contrib.layers.xavier_initializer(),
        activation=self.ACTIVATION
        )
      backward_cell = tf.contrib.rnn.LSTMCell(
        num_units=self.HIDDEN_UNITS,
        use_peepholes = True,
        initializer=tf.contrib.layers.xavier_initializer(),
        activation=self.ACTIVATION
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

    if self.CELL == 'LSTM': 
      # LSTM state is a tuple (c, h) by default
      output_attention = tf.concat(
        [context, output_state_fw[0], output_state_bw[0]], 1)
    else:
      output_attention = tf.concat(
        [context, output_state_fw, output_state_bw], 1)

    output_state = tf.layers.dense(
      inputs=output_attention, 
      units=4*self.HIDDEN_UNITS, 
      activation=self.ACTIVATION,
      use_bias=True, 
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      )

    return output_state, alignments

  def rnn_model_fn(self, features, labels, mode):
    """
    Instantiate a RNN and setup the model for training and inference
    """
    input_layer = tf.unstack(features["data"], axis=2)
    output_state, alignments = self.bidirectional_rnn(input_layer)

    # Classification layer
    output_state = tf.layers.dropout(
      inputs=output_state, rate=0.3, training=mode==tf.estimator.ModeKeys.TRAIN)
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
      global_step = tf.train.get_global_step()
      lrate = tf.train.exponential_decay(self.LEARNING_RATE,
        global_step, 1000, 0.96, staircase=True)
      optimizer = tf.train.AdamOptimizer( 
        learning_rate=lrate,
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

  def train_test(self, dataset, clean_model=True, test_size=0.3):
    """
    Run training procedure and test the model using dataset
    """
    train_data, test_data, train_labels, test_labels = dataset.split_dataset(
      test_size=test_size)
    
    # Clean model directory
    if clean_model:
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

  def crossval(
    self, dataset, train_eval_runs=1, transfer=False, pretrain_dir=''):
    """
    Cross validation for the BRNN
    """
    
    accs, pres, recs, fscs = [], [], [], []

    for k in range(dataset.k):
      time.sleep(60)
      dataset.next_fold()
      train_data    = dataset.train_data        
      train_labels  = dataset.train_labels 
      test_data     = dataset.test_data         
      test_labels   = dataset.test_labels  

      # Clean model directory
      if os.path.isfile(self.MODEL_DIR + '/checkpoint'):
        os.remove(self.MODEL_DIR + '/checkpoint')
      for model_file in glob.glob('/'.join([self.MODEL_DIR, 'model.ckpt*'])):
        os.remove(model_file)

      if transfer == True:
        copy_tree(pretrain_dir, self.MODEL_DIR)

      # Configure GPU memory usage
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True

      # Estimator
      classifier = tf.estimator.Estimator(
        model_fn=self.rnn_model_fn, 
        model_dir=self.MODEL_DIR, 
        config=tf.contrib.learn.RunConfig(
          tf_random_seed=43,save_checkpoints_secs=120*60,session_config=config))
      
      # Train the model computing  metrics over the test set
      train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"data": train_data},
        y=train_labels,
        batch_size=self.BATCH_SIZE,
        num_epochs=None, # Continue until training steps are finished
        shuffle=True
        )
      test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"data": test_data},
        y=test_labels,
        batch_size=self.BATCH_SIZE,
        num_epochs=1, 
        shuffle=False
        )

      if k >= train_eval_runs:
        classifier.train(
          input_fn=train_input_fn,
          max_steps=self.ITERATIONS
          )
      else:
        experiment = tf.contrib.learn.Experiment(
          estimator=classifier,
          train_input_fn=train_input_fn,
          eval_input_fn=test_input_fn,
          train_steps=self.ITERATIONS,
          eval_steps=None, # evaluate runs until input is exhausted
          eval_delay_secs=120, 
          train_steps_per_iteration=1000
          )
        experiment.continuous_train_and_eval() 

      # Test the model and print results
      test_results = classifier.evaluate(input_fn=test_input_fn)
      
      # Compute model metrics with Scikit learn
      predict_labels = []
      predict_results = classifier.predict(input_fn=test_input_fn)
      for prediction in predict_results:
        predict_labels.append(prediction['labels'])
      predict_labels = np.asarray(predict_labels)
      
      acc = test_results['accuracy' ] # Tensorflow
      pre = test_results['precision']
      rec = test_results['recall']
      fsc = 2 * pre * rec / ( pre + rec )
      accs.append(acc); pres.append(pre); recs.append(rec); fscs.append(fsc)

      print("\nCross validation k = %d accuracy = %.5f\n" % (k, acc))
      print(accs); print(pres); print(recs); print(fscs)
    
    print("\n\nTest results\n")
    print('\tacc: mean = %.5f  stdev = %.5f' % (np.mean(accs), np.std(accs))) 
    print('\tpre: mean = %.5f  stdev = %.5f' % (np.mean(pres), np.std(pres))) 
    print('\trec: mean = %.5f  stdev = %.5f' % (np.mean(recs), np.std(recs))) 
    print('\tfsc: mean = %.5f  stdev = %.5f' % (np.mean(fscs), np.std(fscs))) 
    print(accs); print(pres); print(recs);  print(fscs)

if __name__ == "__main__":
  print('Methods module')
