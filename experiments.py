"""
Run experiments for methods.py implementations

"""

import sys, csv, math
import numpy as np
from sklearn import metrics

import methods, datasets, download_datasets

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

if __name__ == "__main__":
  reuters()