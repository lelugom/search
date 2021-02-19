import os, csv, gc, time, shutil, glob, copy, pygtrie
from distutils.dir_util import copy_tree

from time import time
import numpy as np
import tensorflow as tf
import random
from sklearn.utils.linear_assignment_ import linear_assignment
import platform
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec, Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, InputLayer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import metrics

class BahdanauAttention(Layer):
  """
  (Bahdanau et al., 2014) attention mechanism implementation

  [1] https://www.tensorflow.org/tutorials/text/nmt_with_attention
  """
  def __init__(self, units, **kwargs):
    super(BahdanauAttention, self).__init__(**kwargs)
    self.n_units = units
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, inputs):
    (query, values) = inputs
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_length, hidden size)
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class ClusteringLayer(Layer):
  """
  Xifeng Guo, En Zhu, Xinwang Liu, and Jianping Yin. Deep Embedded Clustering with Data Augmentation. ACML 2018.

  [1] https://github.com/XifengGuo/DEC-DA

  Clustering layer converts input sample (feature) to soft label, 
  i.e. a vector that represents the probability of the
  sample belonging to each cluster. The probability is calculated with 
  student's t-distribution.

  # Example
  ```
    model.add(ClusteringLayer(n_clusters=10))
  ```
  # Arguments
    n_clusters: number of clusters.
    weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
    alpha: parameter in Student's t-distribution. Default to 1.0.
    symdec: if SymDEC is required. Default false
  # Input shape
    2D tensor with shape: `(n_samples, n_features)`.
  # Output shape
    2D tensor with shape: `(n_samples, n_clusters)`.
  """

  def __init__(
    self, n_clusters, weights=None, alpha=1.0, symdec=False, **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)
    super(ClusteringLayer, self).__init__(**kwargs)
    self.n_clusters = n_clusters
    self.alpha = alpha
    self.initial_weights = weights
    self.input_spec = InputSpec(ndim=2)

    if symdec == False:
      self.q_fn = lambda inputs : 1.0 / (1.0 + (K.sum(K.square(
        K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / 
        self.alpha))
    else:
      self.q_fn = lambda inputs : 1.0 / (1.0 + ((K.min(K.sqrt(K.sum(
        K.square(K.expand_dims((2*self.clusters)-K.expand_dims(
        inputs, axis=1),axis=1)-K.expand_dims(
        inputs,axis=1)),axis=-1)),axis=1)) / self.alpha))

  def build(self, input_shape):
    assert len(input_shape) == 2
    input_dim = input_shape.as_list()[1]
    self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
    self.clusters = self.add_weight(shape=(self.n_clusters, input_dim),
      initializer='glorot_uniform', name='clusters')
    if self.initial_weights is not None:
      self.set_weights(self.initial_weights)
      del self.initial_weights
    self.built = True

  def call(self, inputs, **kwargs):
    """ 
    student t-distribution, as same as used in t-SNE algorithm.
         q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
    Arguments:
      inputs: the variable containing data, shape=(n_samples, n_features)
    Return:
      q: student's t-distribution, or soft labels for each sample. 
      shape=(n_samples, n_clusters)
    """
    q = self.q_fn(inputs)
    q **= (self.alpha + 1.0) / 2.0
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    return q

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    return input_shape[1], self.n_clusters

  def get_config(self):
    config = {'n_clusters': self.n_clusters}
    base_config = super(ClusteringLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class IRDCS():
  """
  Recurrent Deep Clustering for task identification using a supervised 
  pretrained BiRNN architecture. Use back translation for the clustering phase

  [1] https://www.tensorflow.org/guide/keras/train_and_evaluate
  [2] https://stackoverflow.com/questions/54122211/tensorflow-graph-to-keras-model
  [3] https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
  [4] https://github.com/XifengGuo/DEC-DA
  """
  def __init__(
    self, pretrain_dataset, dataset, model_dir='models/irdcs_model'):
    super(IRDCS, self).__init__()
    self.MODEL_DIR     = model_dir
    self.BATCH_SIZE    = 256
    self.LEARNING_RATE = 1e-5
    self.PRETRAIN_LR   = 1e-4
    self.EPOCHS        = 200    
    self.HIDDEN_UNITS  = 32
    self.NUM_CLASSES   = None
    self.CELL          = 'GRU'
    self.lambda_loss   = 0.1

    self.verbose       = 2
    self.n_clusters    = None
    self.encoder       = None 
    self.classifier    = None
    self.model         = None

    self.dataset = dataset
    self.pretrain_dataset = pretrain_dataset
    self.NUM_CLASSES = len(np.unique(self.pretrain_dataset.labels))
    self.n_clusters =  len(np.unique(self.dataset.labels))

  def build_encoder(self, input_shape):
    """
    Create the recurrent encoder using a bidirectional recurrent layer with
    attention mechanism (Bahdanau et al., 2014) and projection head 
    (Chen et al., 2020) on top of it. 
    Inputs has shape (batch size, number of word vectors, word vector length)

    [1] https://www.tensorflow.org/guide/keras/custom_layers_and_models
    [2] https://github.com/XifengGuo/DEC-DA
    [3] https://www.tensorflow.org/guide/keras/rnn
    """
    units = self.HIDDEN_UNITS
    if self.CELL == 'GRU':
      reccurrent_cell = tf.keras.layers.GRU
    else:
      reccurrent_cell = tf.keras.layers.LSTM

    # Left encoder
    input_layer_left = Input(
      shape=(input_shape[1], input_shape[2]), name='input_left')
    recurrent_layer = tf.keras.layers.Bidirectional(
      layer = reccurrent_cell(
        units=units, return_sequences=True, return_state=True),
      input_shape=input_shape, name='birnn_left')

    if self.CELL == 'GRU':
      rnn_output, fwd_state, bwd_state = recurrent_layer(input_layer_left)
      rnn_state = tf.keras.layers.concatenate([fwd_state, bwd_state])
    else:
      rnn_output, fwd_state_h, fwd_state_c, bwd_state_h, bwd_state_c = \
        recurrent_layer(input_layer_left)
      rnn_state = tf.keras.layers.concatenate(
        [fwd_state_h, fwd_state_c, bwd_state_h, bwd_state_c])

    context, _ = BahdanauAttention(
      units=2*units, name='attention_left')(inputs=(rnn_state, rnn_output))
    attention_output = tf.keras.layers.concatenate([context, rnn_state])

    projection_left = tf.keras.layers.Dense(
      units=16*units, use_bias=True, activation='relu')(attention_output)
    projection_left = tf.keras.layers.Dense(
      units=8*units, use_bias=True, activation='relu')(projection_left)

    # Right encoder
    input_layer_right = Input(
      shape=(input_shape[1], input_shape[2]), name='input_right')
    recurrent_layer = tf.keras.layers.Bidirectional(
      layer = reccurrent_cell(
        units=units, return_sequences=True, return_state=True),
      input_shape=input_shape, name='birnn_right')

    if self.CELL == 'GRU':
      rnn_output, fwd_state, bwd_state = recurrent_layer(input_layer_right)
      rnn_state = tf.keras.layers.concatenate([fwd_state, bwd_state])
    else:
      rnn_output, fwd_state_h, fwd_state_c, bwd_state_h, bwd_state_c = \
        recurrent_layer(input_layer_right)
      rnn_state = tf.keras.layers.concatenate(
        [fwd_state_h, fwd_state_c, bwd_state_h, bwd_state_c])

    context, _ = BahdanauAttention(
      units=2*units, name='attention_right')(inputs=(rnn_state, rnn_output))
    attention_output = tf.keras.layers.concatenate([context, rnn_state])

    projection_right = tf.keras.layers.Dense(
      units=16*units, use_bias=True, activation='relu')(attention_output)
    projection_right = tf.keras.layers.Dense(
      units=8*units, use_bias=True, activation='relu')(projection_right)

    # Classification layers
    projection = tf.keras.layers.concatenate(
      inputs=[projection_left, projection_right])
    classes = tf.keras.layers.Dropout(rate=0.3)(projection)
    classes = tf.keras.layers.Dense(units=1, activation='sigmoid')(classes)

    # Models
    dual_encoder = tf.keras.models.Model(
      inputs=[input_layer_left, input_layer_right], 
      outputs=[projection_left, projection_right], name='dual_encoder')
    classifier = tf.keras.models.Model(
      inputs=[input_layer_left, input_layer_right], 
      outputs=classes, name='classifier')

    return dual_encoder, classifier

  def build_models(self):
    """
    Build the Keras models for encoding, pretraining, and clustering

    [1] https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
    """
    input = self.dataset
    if not os.path.exists(self.MODEL_DIR):
      print('Could not find %s, creating now...' % self.MODEL_DIR)
      os.makedirs(self.MODEL_DIR)
    for model_file in glob.glob('/'.join([self.MODEL_DIR, '*'])):
      if os.path.isfile(model_file):
        os.remove(model_file)
      else:
        shutil.rmtree(model_file)
    tf.keras.backend.clear_session()

    self.encoder, self.classifier = self.build_encoder(input.data.shape)

    reconstruction = tf.keras.layers.Dot(axes=-1, normalize=True)(
      self.encoder.output) 
    clustering_layer = ClusteringLayer(
      self.n_clusters, name="clustering")(self.encoder.output[0])
    self.model = tf.keras.models.Model(
      inputs=self.encoder.input, 
      outputs=[clustering_layer, reconstruction], name='clusters')

    print('\nModel summaries')
    self.encoder.summary()
    self.classifier.summary()
    self.model.summary()

  def pretrain(self, 
    test_size=0.2, loss='binary_crossentropy', metrics=['accuracy']):
    """
    Pretrain the BRNN encoder using pretrain_dataset for a supervised 
    classification problem
    """
    train_data_left, test_data_left, train_data_right, test_data_right, \
      train_labels, test_labels = train_test_split(
      self.pretrain_dataset.data_left, self.pretrain_dataset.data_right,
      self.pretrain_dataset.labels, test_size=test_size)

    self.classifier.compile(
      optimizer=tf.keras.optimizers.Adam(lr=self.PRETRAIN_LR, amsgrad=True),
      loss=loss, 
      metrics=metrics)

    self.classifier.fit([train_data_left, train_data_right], train_labels,
      batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, verbose=self.verbose, 
      validation_data=([test_data_left, test_data_right], test_labels))
    self.classifier.save_weights(self.MODEL_DIR + '/ae_weights.h5')

  def extract_features(self, x, x_aug):
    return self.encoder.predict([x, x_aug])[0]

  def predict(self, x, x_aug):
    q = self.model.predict([x, x_aug], verbose=0)[0]
    return q

  def predict_labels(self, x, x_aug):  
    # predict cluster labels using the output of clustering layer
    return np.argmax(self.predict(x, x_aug), 1)

  @staticmethod
  def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

  def train_on_batch(self, x, y, a, r, sample_weight=None):
    return self.model.train_on_batch([x, a], [y, r], sample_weight)[0]

  def cluster(self, maxiter=2e4, update_interval=8192, tol=1e-3):
    """
    Retrieve embeddings from the pretrained RNN model. Then, run clustering over
    the embedding representation
    """
    x = self.dataset.data
    y = self.dataset.labels
    x_aug = self.dataset.aumgented_data
    adam = tf.keras.optimizers.Adam(lr=self.LEARNING_RATE, amsgrad=True)
    self.model.compile(optimizer=adam, 
      loss=['kld', 'mse'], 
      loss_weights=[self.lambda_loss, 1.0])

    save_interval = int(maxiter)  
    print('Initializing cluster centers with k-means.')
    kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
    features = self.extract_features(x, x_aug)
    y_pred = kmeans.fit_predict(features)
    y_pred_last = np.copy(y_pred)
    self.model.get_layer(name='clustering').set_weights(
      [kmeans.cluster_centers_])
    
    logfile = open(self.MODEL_DIR + '/log.csv', 'w')
    logwriter = csv.DictWriter(
      logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
    logwriter.writeheader()

    loss = 0
    for ite in range(int(maxiter)): 
      q = self.predict(x, x_aug)
      epoch = (x.shape[0] // self.BATCH_SIZE) * 1
      if ite % epoch == 0:
        p = self.target_distribution(q) 

      y_pred = q.argmax(1)
      avg_loss = loss / update_interval
      loss = 0.
      if y is not None:
        acc = np.round(metrics.acc(y, y_pred), 5)
        nmi = np.round(metrics.nmi(y, y_pred), 5)
        ari = np.round(metrics.ari(y, y_pred), 5)
        logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=avg_loss)
        logwriter.writerow(logdict)
        logfile.flush()
        print('Iter %d: acc=%.5f, nmi=%.5f, ari=%.5f; loss=%.5f' % (
          ite, acc, nmi, ari, avg_loss))

      delta_label = np.sum(
        y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
      y_pred_last = np.copy(y_pred)

      idx = np.random.randint(0, x.shape[0], self.BATCH_SIZE)
      ones = np.ones(idx.shape[0], dtype=np.float32)
      loss += self.train_on_batch(x=x[idx], y=p[idx], a=x_aug[idx], r=ones)

    logfile.close()
    print('saving model to:', self.MODEL_DIR + '/model_final.h5')
    self.model.save_weights(self.MODEL_DIR + '/model_final.h5')

    return y_pred
