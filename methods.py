"""
Sesarch session segmentation using a number of methods
"""

import download_datasets, datasets

import os, gc, time, shutil, glob, copy, distance, multiprocessing
import ngtpy
from distutils.dir_util import copy_tree

import numpy as np
import tensorflow as tf
import networkx as nx

from tensorflow.keras.optimizers import SGD, Adam
from keras.initializers import VarianceScaling
from scipy.spatial.distance import cosine

from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, k_means_, DBSCAN, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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

def pairwise_fscore(true_labels, predicted_labels):
  _, fn, fp, tp = pairwise_counts(true_labels, predicted_labels)

  precision = tp / (tp + fp + 1e-10)
  recall = tp / (tp + fn + 1e-10)
  pairwise_fscore = 2 * precision * recall / (precision + recall + 1e-10)

  return pairwise_fscore, precision, recall

def get_num_clusters(labels):
  """ Compute the number of clusters from the dataset labels """
  num_clusters = np.max(labels)
  if np.min(labels) == 0:
    num_clusters += 1
  return num_clusters

class mgbc(object):
  """
  Clustering method using multilingual sentence embeddings and angular 
  distance (Yang et al., 2018) for graphs
  """
  def __init__(self, data, labels, threshold=-0.3, alpha=0.4, 
    representation=datasets.representation().glove, 
    semantic=None):
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
    Compute similarity between word embeddings using angular distance. 
    Return negative angle
    """
    semantic_sim = 0
    if isinstance(q0, str):
      lexical = self.compute_representation([q0, q1])
      if self.compute_semantic != None:
        semantic_sim = self.compute_semantic(q0, q1)
      q0, q1 = lexical[0], lexical[1]

    cos = np.dot(q0,q1) / (np.linalg.norm(q0)+1e-8) / (np.linalg.norm(q1)+1e-8)
    angle = np.arccos(np.clip(cos, -1.0, 1.0))
    lexical_sim = -angle
    sim = self.alpha * lexical_sim + (1 - self.alpha) * semantic_sim
    return sim

  def build_graph(self):
    """
    Use query indexes as nodes and query distances as weights
    """
    for i in range(len(self.data) - 1):
      for j in range(i + 1, len(self.data)):
        weight = self.query_similarity(self.data[i], self.data[j])
        self.graph.add_edge(i, j, weight=weight)

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
    Run graph based method for queries clustering
    """
    self.build_graph()
    self.prune_graph()
    self.label_queries()

    fscore, _, _ = pairwise_fscore(self.labels, self.predicted_labels)
    print('\nthreshold %.2f alpha %.2f fscore %.3f'%(
      -self.threshold, self.alpha, fscore))

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
    
    print("\n\nTest results\n")
    print('\tacc: mean = %.5f  stdev = %.5f' % (np.mean(accs), np.std(accs))) 
    print('\tfsc: mean = %.5f  stdev = %.5f' % (np.mean(fscs), np.std(fscs))) 

class multiplelearners(object):
  """
  Run scikit learn classifiers
  """

  def __init__(self):
    self.MODEL_DIR     = 'models/multiplelearners'
    self.classifiers = [
      LogisticRegression(),
      KNeighborsClassifier(3),
      SVC(kernel="linear", C=0.025),
      SVC(gamma=2, C=1),
      GaussianProcessClassifier(1.0 * RBF(1.0)),
      DecisionTreeClassifier(max_depth=5),
      RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
      MLPClassifier(alpha=1, max_iter=1000),
      AdaBoostClassifier(),
      GaussianNB(),
      QuadraticDiscriminantAnalysis()]
    self.names = [
      "Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM", 
      "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]

  def colapse(self,nar):
    """ 
    Convert input samples into a 1-dimensional vector
    """
    sn = nar.shape
    if len(sn) > 2:
      return nar.reshape(sn[0],sn[1]*sn[2])
    return nar

  def crossval(
    self, dataset, pretrain_dir='', iclass=0, metrics_average=False):
    """
    Cross validation for scikit learn classifiers
    """

    name = self.names[iclass]
    accs, pres, recs, fscs = [], [], [], []
    print("\n ",name," ")

    for _ in range(dataset.k):
      dataset.next_fold()
      train_data    = self.colapse(dataset.train_data)        
      train_labels  = dataset.train_labels 
      test_data     = self.colapse(dataset.test_data)         
      test_labels   = dataset.test_labels  

      clf = self.classifiers[iclass]
      clf.fit(train_data, train_labels)
      predict_labels = clf.predict(test_data)
      
      # Compute model metrics with Scikit learn      
      acc = metrics.accuracy_score(test_labels, predict_labels)
      if metrics_average:
        pre = metrics.precision_score(
          test_labels, predict_labels, average='macro')
        rec = metrics.recall_score(
          test_labels, predict_labels, average='macro')
      else:
        pre = metrics.precision_score(test_labels, predict_labels)
        rec = metrics.recall_score(test_labels, predict_labels)
      fsc = 2 * pre * rec / ( pre + rec + 1e-10)
      accs.append(acc); pres.append(pre); recs.append(rec); fscs.append(fsc)
      del clf

    print("\n\nTest results\n")
    print('\tacc: %.3f +/- %.3f' % (np.mean(accs), np.std(accs))) 
    print('\tfsc: %.3f +/- %.3f' % (np.mean(fscs), np.std(fscs))) 

  def run(self, dataset, metrics_average=False):
    """ 
    Run crossvalidation for all the classifiers using dataset. metrics_average
    should be False for binary classification runs. For multiclass runs, set 
    metrics_average=True. This will ensure sklearn gets the 'macro' option for
    average in precision and recall method calls
    """
    for i in range(len(self.names)):
      dataset.kfold()
      self.crossval(dataset=dataset, iclass=i, metrics_average=metrics_average)

class task_rules(object):
  """
  Search segmentation using the heuristics based approach in 
  (Gomes et al., 2019)

  [1] https://github.com/PedroG1515/Segmenting-User-Sessions
  """
  def __init__(self, dataset):
    self.dataset = dataset
    self.representation = datasets.representation()
    self.compute_representation = self.representation.glove

  def ngrams(self, q, n=3):
    """
    Character n-gram for a query sentence
    """
    ngrams = []
    tokens = q.split()
    for token in tokens:
      for i in range(0, len(token) - n - 1):
        ngram = token[i : i + n]
        ngrams.append(str(ngram))
    return ngrams

  def jaccard_n(self, q0, q1, n=3):
    """
    Jaccard similarity between character n-grams of two strings

    [1] https://github.com/doukremt/distance
    """
    ngrams0 = self.ngrams(q0, n=n)
    ngrams1 = self.ngrams(q1, n=n)
    if len(ngrams0) + len(ngrams1) == 0:
      return 0.0
    return 1.0 - distance.jaccard(ngrams0, ngrams1)

  def jaccard_mn(self, q0, q1, m= 3, n=4):
    """
    Jaccard similarity between character m-grams and n-grams of two strings

    [1] https://github.com/doukremt/distance
    """
    mngrams0 = self.ngrams(q0, n=m)
    mngrams0.extend(self.ngrams(q0, n=n))
    mngrams1 = self.ngrams(q1, n=m)
    mngrams1.extend(self.ngrams(q1, n=n))
    if len(mngrams0) + len(mngrams1) == 0:
      return 0.0
    return 1.0 - distance.jaccard(mngrams0, mngrams1)

  def cosine(self, q0, q1):
    """
    Cosine similarity of average word embeddings
    """
    if isinstance(q0, str):
      lexical = self.compute_representation([q0, q1])
      q0, q1 = lexical[0], lexical[1]

    cos = np.dot(q0,q1) / (np.linalg.norm(q0)+1e-8) / (np.linalg.norm(q1)+1e-8)
    return cos

  def word_movers_distance(self, q0, q1):
    """
    Word movers distance from sets of embeddings representing q0 and q1
    """
    return self.representation.vectors.wmdistance(q0, q1)

  def temporal_component(self, time_gap):
    """
    Temporal component heuristic to assess the time gap in seconds between 
    a pair of queries
    """
    return max(
      0, 1.0 - time_gap / min(24.0 * 60.0 * 60.0, 2 * self.dataset.max_gap))

  def extract_url_domain_name(self, url):
    """
    https://github.com/PedroG1515/Segmenting-User-Sessions/blob/master/utils.py
    """
    from urllib.parse import urlparse
    domain = urlparse(url).hostname
    return domain.split('.')[1]

  def size_lcs(self, X, Y):
    """
    https://github.com/PedroG1515/Segmenting-User-Sessions/blob/master/utils.py
    """
    m = len(X)
    n = len(Y)
    L = [[None]*(n + 1) for i in range(m + 1)]
    for i in range(m + 1):
      for j in range(n + 1):
        if i == 0 or j == 0:
          L[i][j] = 0
        elif X[i-1] == Y[j-1]:
          L[i][j] = L[i-1][j-1]+1
        else:
          L[i][j] = max(L[i-1][j], L[i][j-1])

    return L[m][n]

  def url_similarity(self, url_row, url_log):
    """
    https://github.com/PedroG1515/Segmenting-User-Sessions/blob/master/Proposed_Method_with_WMD.py
    """
    url_row_filter = self.extract_url_domain_name(url_row.lower())
    url_log_filter = self.extract_url_domain_name(url_log.lower())
    lennsubstring = self.size_lcs(url_row_filter, url_log_filter)
    lennn = len(url_row_filter)
    if len(url_log_filter) > lennn:
        lennn = len(url_log_filter)
    simi_sub = lennsubstring/lennn

    if url_row_filter == '' or url_log_filter == '' :
      simi_sub = 0

    return simi_sub

  def predict(self):
    """
    Predict if pairs of queries in self.dataset.data are part of the same 
    task or not
    """
    predicted_labels = []
    for data in self.dataset.data:
      q0, q1, time_gap, url0, url1 = data[0], data[1], data[2], data[3], data[4]
      ft = self.temporal_component(time_gap)
      fl1 = self.jaccard_n(q0, q1)
      if fl1 > np.sqrt(1 - ft * ft):
        predicted_labels.append(0)
      else:
        fl2 = self.jaccard_mn(q0, q1)
        if np.sqrt(ft * ft + fl2 * fl2) > 1:
          predicted_labels.append(0)
        else:
          if ft > 0.7 and fl2 < 0.5:
            fs1 = self.cosine(q0, q1)
            if fs1 > 0.5:
              predicted_labels.append(0)
            else:
              fs2 = self.word_movers_distance(q0, q1)
              if fs2 < 0.1:
                predicted_labels.append(0)
              else:
                if len(url0) > 0 and len(url1) > 0 and \
                  np.sqrt(fs1 * fs1 + fs2 * fs2) > 1:
                  fu = self.url_similarity(url0, url1)
                  if fu > 0.7:
                    predicted_labels.append(0)
                  else:
                    predicted_labels.append(1)
                else:
                  predicted_labels.append(1)
          else:
            predicted_labels.append(1)
    return predicted_labels

  def test(self):
    """
    Test the segmentation method using self.dataset. Report metrics accordingly
    """
    predict_labels = self.predict()
    test_labels = self.dataset.labels

    acc = metrics.accuracy_score(test_labels, predict_labels)
    pre = metrics.precision_score(test_labels, predict_labels)
    rec = metrics.recall_score(test_labels, predict_labels)
    fsc = 2 * pre * rec / ( pre + rec + 1e-10)

    print("\n\nTest results\n")
    print('\tacc: %.3f +/- %.3f' % (np.mean(acc), np.std(acc))) 
    print('\tfsc: %.3f +/- %.3f' % (np.mean(fsc), np.std(fsc))) 

class task_map(object):
  """
  Query task mapping using NGT. Testing parameters from  (Volske et al., 2019). 
  """
  def __init__(self, dataset):
    super(task_map, self).__init__()
    self.dataset = dataset
    self.repetitions = 100
    self.test_time_samples = 10000
    self.test_time_ms = None
    self.runs = 50
    self.annoy_metric = 'angular'
    self.annoy_n = 1
    self.ngt_distance = 'Normalized Angle'

  def ngt_predict(self, model, test_data, train_labels):
    """
    Predict labels using an NGT index structure
    """
    predicted_labels = []
    for test in test_data:
      results = model.search(test, self.annoy_n)
      votes = {}
      for result in results:
        index = result[0]
        label = train_labels[index]
        votes[label] = votes.get(label, 0) + 1
      max_vote, max_count = 0, 0
      for label in votes.keys():
        if votes[label] > max_count:
          max_count = votes[label]
          max_vote = label
      predicted_labels.append(max_vote)
    return predicted_labels

  def ngt(self, train_data, train_labels, test_data, test_labels):
    """
    Run NGT. Compute training and testing times

    [1] https://github.com/yahoojapan/NGT
    """
    train_time = 0
    time0 = time.time()
    
    ngtpy.create(b"data", len(train_data[0]), distance_type=self.ngt_distance)
    model = ngtpy.Index(b"data")
    model.batch_insert(train_data)
    time1 = time.time()
    model.save()
    predicted_labels = self.ngt_predict(model, test_data, train_labels)

    if self.test_time_ms == None:
      time2 = time.time()
      self.ngt_predict(
        model, train_data[0:self.test_time_samples], train_labels)
      time3 = time.time()
      self.test_time_ms = 1000.0 * (time3 - time2) / (self.test_time_samples)

    train_time = time1 - time0
    return predicted_labels, train_time

  def map_ngt(self):
    """
    Task mapping using NGT
    """
    self.test(self.ngt)

  def test(self, method):
    """
    Test the task mapping method using the number of self.runs specified
    """
    train_times = []
    self.test_time_ms = None
    accs, recs, pres, fscs = [], [], [], []

    for i in range(self.runs):
      print('\trun %d' % i, end=' ', flush=True)
      predicted, expected, train_t = self.leave_one_out(method)
      train_times.extend(train_t)
      acc = metrics.accuracy_score(expected, predicted)
      pre = metrics.precision_score(expected, predicted, average='macro')
      rec = metrics.recall_score(expected, predicted, average='macro')
      fsc = 2 * pre * rec / ( pre + rec + 1e-10)
      accs.append(acc)
      pres.append(pre)
      recs.append(rec)
      fscs.append(fsc)

    results =  ("\n\nTest results\n")
    results += ('\tacc: %.3f +/- %.3f' % (np.mean(accs), np.std(accs))) 
    results += ('\tfsc: %.3f +/- %.3f' % (np.mean(fscs), np.std(fscs))) 
    results += ('\ttest_t (ms): %.3f +/- %.3f' % (
      self.test_time_ms, 0.0)) 
    print(results)

  def leave_one_out(self, method):
    """
    Randomly select one sample for test. Train the model with the remaining 
    samples. 
    """
    train_times = []
    predicted_labels = np.asarray([], dtype=np.int32)
    expected_labels  = np.asarray([], dtype=np.int32)
    dataset_size = self.dataset.data.shape[0]
    outs = np.random.randint(dataset_size, size=self.repetitions)

    for out in outs:
      train_idxs = np.delete(np.arange(dataset_size), out)
      test_idxs = np.asarray([out], dtype=np.int32)
      train_data = self.dataset.data[train_idxs]
      train_labels = self.dataset.labels[train_idxs]
      test_data = self.dataset.data[test_idxs]
      test_labels = self.dataset.labels[test_idxs]
      expected_labels = np.append(expected_labels, test_labels)
      predicted_label, train_time = method(
        train_data, train_labels, test_data, test_labels)
      predicted_labels = np.append(predicted_labels, predicted_label)
      train_times.append(train_time)

    return predicted_labels, expected_labels, train_times
