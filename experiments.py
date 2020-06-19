 
"""
Run experiments for methods.py implementations
"""

import sys, csv, math
import numpy as np
from sklearn import metrics

import methods, datasets, download_datasets

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

def task_seg_crossval_brnn():
  """ 
  Run a experiment with the bidirectional RNN to detect session changes in
  the AOL dataset from (Sen et al., 2018) and (Hagen et al., 2013)
  """

  m = 0
  hagen_aol=datasets.hagen_aol(representation=datasets.representation().glove)
  hagen_aol.load_sequential_queries(m=m, n=m+1)
  hagen_aol.kfold()

  brnn = methods.brnn()
  brnn.MODEL_DIR  = 'models/rnn_model_transfer'
  brnn.ITERATIONS = 60000
  brnn.CELL = 'LSTM'
  brnn.BATCH_SIZE = 256
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + '. Hagen et al. dataset. m,n= ' + str(m) +',' + str(m+1)
  print(description, file=sys.stderr)
  print(description)
  brnn.crossval(hagen_aol)
  del hagen_aol

  m = 0
  sen_aol=datasets.lucchese_aol(representation=datasets.representation().glove)
  sen_aol.load_sequential_queries(m=m, n=m+1)
  sen_aol.kfold()

  brnn = methods.brnn()
  brnn.MODEL_DIR  = 'models/rnn_model_transfer'
  brnn.ITERATIONS = 20000
  brnn.CELL = 'LSTM'
  brnn.BATCH_SIZE = 256
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + '. Sen et al. dataset. m,n= ' + str(m) +',' + str(m+1)
  print(description, file=sys.stderr)
  print(description)
  brnn.crossval(sen_aol, train_eval_runs=1)
  del sen_aol

  m = 0
  hagen_aol=datasets.hagen_aol(representation=datasets.representation().glove)
  hagen_aol.load_sequential_queries(m=m, n=m+1)
  hagen_aol.kfold()

  brnn = methods.brnn()
  brnn.MODEL_DIR  = 'models/rnn_model_transfer'
  brnn.ITERATIONS = 60000
  brnn.CELL = 'GRU'
  brnn.BATCH_SIZE = 256
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + '. Hagen et al. dataset. m,n= ' + str(m) +',' + str(m+1)
  print(description, file=sys.stderr)
  print(description)
  brnn.crossval(hagen_aol)
  del hagen_aol

  m = 0
  sen_aol=datasets.lucchese_aol(representation=datasets.representation().glove)
  sen_aol.load_sequential_queries(m=m, n=m+1)
  sen_aol.kfold()

  brnn = methods.brnn()
  brnn.MODEL_DIR  = 'models/rnn_model_transfer'
  brnn.ITERATIONS = 20000
  brnn.CELL = 'GRU'
  brnn.BATCH_SIZE = 256
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + '. Sen et al. dataset. m,n= ' + str(m) +',' + str(m+1)
  print(description, file=sys.stderr)
  print(description)
  brnn.crossval(sen_aol, train_eval_runs=1)
  del sen_aol

def task_seg_context():
  """
  Use adjacent queries to add context to the query pair. Use Hagen et al., 2013
  dataset and the BRNN architecture with GRU cells
  """

  for m in range(0, 6):
    hagen_aol = datasets.hagen_aol(
      representation=datasets.representation().glove)
    hagen_aol.load_sequential_queries(m=m, n=m+1)
    hagen_aol.kfold()

    brnn = methods.brnn()
    brnn.MODEL_DIR  = 'models/rnn_model_context'
    brnn.ITERATIONS = 60000
    brnn.CELL = 'GRU'
    brnn.BATCH_SIZE = 256
    description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
      brnn.CELL + '. Hagen et al. dataset. m,n= ' + str(m) +',' + str(m+1)
    print(description, file=sys.stderr)
    print(description)
    brnn.crossval(hagen_aol)
    del hagen_aol

    if m == 0:
      continue
      
    hagen_aol = datasets.hagen_aol(
      representation=datasets.representation().glove)
    hagen_aol.load_sequential_queries(m=m, n=1)
    hagen_aol.kfold()

    brnn = methods.brnn()
    brnn.MODEL_DIR  = 'models/rnn_model_context'
    brnn.ITERATIONS = 60000
    brnn.CELL = 'GRU'
    brnn.BATCH_SIZE = 256
    description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
      brnn.CELL + '. Hagen et al. dataset. m,n= ' + str(m) +',' + str(1)
    print(description, file=sys.stderr)
    print(description)
    brnn.crossval(hagen_aol)
    del hagen_aol

def task_seg_transfer_learning():
  """
  Pretrain BRNN on Hagen et al., 2013 dataset. Then, fine tune for Sen et al., 2018 session segmentation
  """

  model_dir = 'models/rnn_model'
  pretrain_model_dir = 'models/rnn_model_transfer_pretrain' 
  
  hagen_aol = datasets.hagen_aol(
    representation=datasets.representation().glove)
  hagen_aol.load_sequential_queries()
  brnn = methods.brnn()
  brnn.MODEL_DIR  = pretrain_model_dir
  brnn.ITERATIONS = 40000
  brnn.CELL = 'GRU'
  brnn.BATCH_SIZE = 256 
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + \
    '. Sen et al. dataset. Pretraining with Hagen et al., test set 10%' 
  print(description, file=sys.stderr)
  print(description)
  brnn.train_test(hagen_aol, test_size=0.1)
  del hagen_aol

  sen_aol = datasets.lucchese_aol(
    representation=datasets.representation().glove)
  sen_aol.load_sequential_queries()
  sen_aol.kfold()
  brnn = methods.brnn()
  brnn.MODEL_DIR  = model_dir
  brnn.ITERATIONS = 40000 + 20000
  brnn.CELL = 'GRU'
  brnn.BATCH_SIZE = 256 
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + '. Sen et al. dataset, crossval, fine tuning' 
  print(description, file=sys.stderr)
  print(description)
  brnn.crossval(sen_aol, transfer=True, pretrain_dir=pretrain_model_dir)
  del sen_aol

  hagen_aol = datasets.hagen_aol(
    representation=datasets.representation().glove)
  hagen_aol.load_sequential_queries()
  brnn = methods.brnn()
  brnn.MODEL_DIR  = pretrain_model_dir
  brnn.ITERATIONS = 40000
  brnn.CELL = 'LSTM'
  brnn.BATCH_SIZE = 256 
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + \
    '. Sen et al. dataset. Pretraining with Hagen et al., test set 10%' 
  print(description, file=sys.stderr)
  print(description)
  brnn.train_test(hagen_aol, test_size=0.1)
  del hagen_aol

  sen_aol = datasets.lucchese_aol(
    representation=datasets.representation().glove)
  sen_aol.load_sequential_queries()
  sen_aol.kfold()
  brnn = methods.brnn()
  brnn.MODEL_DIR  = model_dir
  brnn.ITERATIONS = 40000 + 20000
  brnn.CELL = 'LSTM'
  brnn.BATCH_SIZE = 256 
  description = '\n\n--- Experiment with ' + str(brnn.HIDDEN_UNITS) + \
    brnn.CELL + '. Sen et al. dataset, crossval, fine tuning' 
  print(description, file=sys.stderr)
  print(description)
  brnn.crossval(sen_aol, transfer=True, pretrain_dir=pretrain_model_dir)
  del sen_aol

def task_seg_ml_sklearn_crossval():
  """
  Run ML Scikit learn methods for Sen et al. and Hagen et al. dataset task 
  segmentation
  """
  classifiers = methods.multiplelearners()

  print('\n\n--- Experiment with Sen et al. dataset ')
  sen_aol = datasets.lucchese_aol(
    representation=datasets.representation().glove)
  sen_aol.load_sequential_queries()
  classifiers.run(sen_aol)

  dataset = datasets.lucchese_aol()
  dataset.load_sequential_pair()
  methods.task_rules(dataset=dataset).test()

  print('\n\n--- Experiment with Hagen et al. dataset ')
  hagen_aol = datasets.hagen_aol(
    representation=datasets.representation().glove)
  hagen_aol.load_sequential_queries()
  classifiers.run(hagen_aol)
  
  dataset = datasets.hagen_aol()
  dataset.load_sequential_pair()
  methods.task_rules(dataset=dataset).test()

def task_seg():
  """
  Replicate results for task segmentation experiemnts
  """
  task_seg_crossval_brnn()
  task_seg_ml_sklearn_crossval()
  task_seg_transfer_learning()
  task_seg_context()

def mgbc_task_ide(clueweb_url):
  """
  Graph based clustering for search task identification
  """
  representation = datasets.representation(lang='').universal_sentence_encoder
  dataset = datasets.sen_aol(representation=representation)

  if clueweb_url == '':
    dataset.load(textdata=False)
    semantic = None
  else:
    dataset.load(textdata=True)
    clueweb = datasets.clueweb()
    clueweb.BASE_URL = clueweb_url + '?query='
    semantic = clueweb.semantic_similarity_ids

  alphas = [alpha for alpha in np.arange(0.1, 1.01, 0.1)]
  thresholds = [-threshold for threshold in np.arange(0.1, 1.01, 0.1)]
  for threshold in thresholds:
    for alpha in alphas:
      mgbc = methods.mgbc(
        dataset.data, dataset.labels, threshold=threshold, 
        alpha=alpha, representation=representation, semantic=semantic)
      mgbc.cluster()

def ngt_task_map():
  """
  Run task mapping experiments on query task mapping datasets 
  (Volske et al., 2019).
  """
  print('\n\n--- Experiments with NGT')
  representation = datasets.representation(lang='').universal_sentence_encoder
  volske = datasets.volske_aol(representation=representation)
  volske.load(textdata=False)
  volske.data = np.asarray(volske.data)
  task_map = methods.task_map(dataset=volske)
  task_map.map_ngt()
  
  volske = datasets.volske_trek(representation=representation)
  volske.load(textdata=False)
  volske.data = np.asarray(volske.data)
  task_map = methods.task_map(dataset=volske)
  task_map.map_ngt()

  volske = datasets.volske_wikihow(representation=representation)
  volske.load(textdata=False)
  volske.data = np.asarray(volske.data)
  task_map = methods.task_map(dataset=volske)
  task_map.map_ngt()

def task_ide(clueweb_url=''):
  """
  Replicate results for task identification experiments
  """
  mgbc_task_ide(clueweb_url)
  ngt_task_map()

if __name__ == "__main__":
  print('checking datasets ...')
  download_datasets.download_files()
  
  if len(sys.argv) == 3 and sys.argv[2] == 'segmentation':
    print('running search segmentation experiments ...')
    task_seg()
  elif len(sys.argv) == 3 and sys.argv[2] == 'identification':
    print('running task identification experiments ...')
    task_ide()
  elif len(sys.argv) == 5 and sys.argv[2] == 'identification':
    print('running task identification experiments ...')
    url = sys.argv[4] + '?query='
    task_ide(clueweb_url=url)
  else:
    print("""
      Usage:
      python experiments.py -t segmentation
      python experiments.py -t identification
      python experiments.py -t identification -u clueweb_url
      """)