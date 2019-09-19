 
"""
Run experiments for methods.py implementations
"""

import sys, csv, math
import numpy as np
from sklearn import metrics

import methods, datasets, download_datasets

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

def reuters(runs=10):
  """
  Use 10k samples from Reuters dataset to run DEC, IDEC, and SymDEC. 
  """

  print('\n\n--- Experiment with reuters dataset')
  reuters = datasets.reuters()
  reuters.load()
  
  methods.kmeans(reuters.data, reuters.labels)
  methods.dbscan(reuters.data, reuters.labels)
  methods.aglomerative_clustering(reuters.data, reuters.labels)
  methods.decs(reuters, 'reuters', runs=runs, pretrain_epochs=15, update_interval=30, maxiter=150, tol=0.001)

def crossval_brnn_aol():
  """ 
  Run a experiment with the bidirectional RNN to detect session changes in
  the AOL dataset from (Lucchese et al., 2011) and (GayoAvello et al., 2006)
  """

  m = 0
  gayo = datasets.aol_gayo(
    representation=datasets.representation().fastText)
  gayo.load_sequential_queries(m=m, n=m+1)
  gayo.kfold()

  brnn = methods.brnn()
  brnn.CELL = 'LSTM'
  brnn.BATCH_SIZE = 256
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + '. GayoAvello et al. dataset. m,n= ' + str(m) +',' + str(m+1)
  print(description, file=sys.stderr)
  print(description, flush=True)
  brnn.crossval(gayo, train_eval_runs=gayo.k)

def transfer_learning():
  """
  Pretrain BRNN on GayoAvello et al., 2006 dataset. Then, fine tune for Sen et al., 2018 session segmentation
  """

  model_dir = 'models/rnn_model_transfer'
  pretrain_model_dir = 'models/rnn_model_transfer_pretrain'
  procheta = datasets.aol_lucchese(
    representation=datasets.representation().fastText)
  procheta.load_sequential_pair()
  procheta.kfold()
  brnn = methods.brnn()
  brnn.MODEL_DIR  = model_dir
  brnn.ITERATIONS = 20000
  brnn.CELL = 'LSTM'
  brnn.BATCH_SIZE = 256 
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + '. Sen et al. dataset, test set 10%' 
  print(description, file=sys.stderr)
  print(description, flush=True)
  brnn.crossval(procheta, transfer=False)
  del procheta

  gayo = datasets.aol_gayo(
    representation=datasets.representation().fastText)
  gayo.load_sequential_pair()
  brnn = methods.brnn()
  brnn.MODEL_DIR  = pretrain_model_dir
  brnn.ITERATIONS = 40000
  brnn.CELL = 'LSTM'
  brnn.BATCH_SIZE = 256 
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + \
    '. Sen et al. dataset. Pretraining with GayoAvello et al., test set 10%' 
  print(description, file=sys.stderr)
  print(description, flush=True)
  brnn.train_test(gayo, test_size=0.1)
  del gayo

  procheta = datasets.aol_lucchese(
    representation=datasets.representation().fastText)
  procheta.load_sequential_pair()
  procheta.kfold()
  brnn = methods.brnn()
  brnn.MODEL_DIR  = model_dir
  brnn.ITERATIONS = 40000 + 20000
  brnn.CELL = 'LSTM'
  brnn.BATCH_SIZE = 256 
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + '. Sen et al. dataset fine tuning, test set 10%' 
  print(description, file=sys.stderr)
  print(description, flush=True)
  brnn.crossval(procheta, transfer=True, pretrain_dir=pretrain_model_dir)

if __name__ == "__main__":
  reuters()
  crossval_brnn_aol()
  transfer_learning()
