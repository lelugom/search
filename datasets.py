"""
Load data from local folder and encode information according to the selected
representation
"""

import download_datasets

import io, csv, os, sys, distance
import numpy as np 

from datetime import timezone, datetime

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.test.utils import get_tmpfile
from gensim.models.fasttext import load_facebook_vectors
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Print the whole arrays and set random seed
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(43)

class representation(object):
  """
  Implement different representation approaches to encode text data
  """

  def minhash(self, text_data):
    """
    Use the minhash algorithm to encode text_data, an array of strings. 
    Returns a vector for every string

    [1] https://github.com/go2starr/lshhdc
    [2] https://stackoverflow.com/questions/14533420/can-you-suggest-a-good-minhash-implementation
    """
    signatures = []
    MINHASH_WIDTH = 128

    for line in text_data:
      words = line.split()
      signature = [float('inf')] * MINHASH_WIDTH
      for i in range(MINHASH_WIDTH):
        signature[i] = min([
          hash("salt" + str(i) + str(word) + "salt") 
          for word in words])
      
      signature = np.array(signature) / max(signature)
      signatures.append(signature)

    return signatures

  def tfidf(self, text_data):
    """
    Execute CountVectorizer to tokenize text data, then compute the TF
    IDF for the data representation

    [1] https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    """

    tokens = CountVectorizer(dtype=np.float32).fit_transform(text_data)
    frequencies = TfidfTransformer(
      norm='l2', sublinear_tf=True).fit_transform(tokens)
    frequencies = np.asarray(
      frequencies.todense()) * np.sqrt(frequencies.shape[1])

    return frequencies

  def fastText(self, text_data):
    """
    Pretrained word embeddings from fastText

    [1] https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
    [2] https://radimrehurek.com/gensim/models/fasttext.html
    """

    model = load_facebook_vectors('datasets/fasttext/cc.en.300.bin')
    vectors = model
    return self.compute_avg_embeddings(text_data, vectors)

  def word2vec(self, text_data):
    """
    Pretrained word embeddings from word2vec

    [1] https://code.google.com/archive/p/word2vec/
    """

    vectors = KeyedVectors.load_word2vec_format(
      'datasets/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    return self.compute_embeddings(text_data, vectors)
    
  def glove(self, text_data):
    """
    Pretrained word embeddings from GloVe

    [1] https://nlp.stanford.edu/projects/glove/
    [2] https://radimrehurek.com/gensim/scripts/glove2word2vec.html
    """

    glove2word2vec('datasets/glove/glove.42B.300d.txt', 'datasets/glove/glove.42B.300d_word2vec.txt')

    vectors = KeyedVectors.load_word2vec_format('datasets/glove/glove.42B.300d_word2vec.txt')
    return self.compute_embeddings(text_data, vectors)

  def compute_embeddings(self, text_data, vectors):
    """
    Calculate embeddings for every sentence in text_data array. Use the 
    model stored in vectors to retrieve pretrained word embeddings
    """

    embedding_length = len(vectors['tree'])
    query_lengths = []
    for query in text_data:
      query_lengths.append(len(query.split()))
    query_length = int(np.std(query_lengths) + 2 * np.mean(query_lengths))

    embeddings = []
    for query in text_data:
      not_in_vocabulary = [0.0 for i in range(embedding_length)]
      embedding = np.zeros((query_length, embedding_length))

      for index, word in enumerate(query.split()):
        if index == query_length:
          break

        try:
          vector = vectors[word]
        except:
          vector = not_in_vocabulary
          print('\t  ' + word + ' not in vocabulary')
        embedding[index] = np.asarray(vector, dtype=np.float32)

      embeddings.append(embedding)

    return embeddings

  def compute_avg_embeddings(self, text_data, vectors):
    """
    Calculate embeddings for every sentence in text_data array. Use the 
    model stored in vectors to retrieve pretrained word embeddings and compute
    the average of word vectors
    """

    embedding_length = len(vectors['tree'])
    embeddings = []

    for query in text_data:
      not_in_vocabulary = [0.0 for i in range(embedding_length)]
      embedding = []
      
      for _, word in enumerate(query.split()):
        try:
          vector = vectors[word]
        except:
          vector = not_in_vocabulary
          print('\t  ' + word + ' not in vocabulary')
        embedding.append(vector)

      embedding = np.asarray(embedding, dtype=np.float32)
      embeddings.append(np.mean(embedding, axis=0))

    return embeddings

class mnist(object):
  """
  Popular image dataset for testing purposes

  [1] https://github.com/XifengGuo/IDEC
  """
  def __init__(self):
    self.data = []
    self.labels = []

  def load(self):
    """
    Take data from keras 
    """

    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 50.) # normalization as DEC
    
    self.data = x
    self.labels = y

class iris(object):
  """
  Popular flowers dataset for testing purposes

  [1] https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py
  """
  def __init__(self):
    self.data = []
    self.labels = []

  def load(self):
    """
    Take data from scikit learn 
    """

    from sklearn import datasets
    np.random.seed(5)
    iris = datasets.load_iris()
    self.data   = iris.data
    self.labels = iris.target

class aol_procheta(object):
  """
  Search session task dataset from (Sen et al., 2018)
  """

  def __init__(self, representation=representation().minhash):
    self.file = '/'.join(
      [download_datasets.DATA_DIR, 'aol', 'procheta_task.csv'])

    self.data    = []
    self.labels  = []
    self.scaler  = None

    self.compute_representation = representation

  def load(self, textdata=False):
    """
    Load data into object variables according to the specified representation
    """

    with open(self.file, mode='r') as data_file:
      reader = csv.reader(data_file, delimiter=',')
      for row in reader:
        self.data.append(row[0].strip())
        self.labels.append(int(row[1]))

    if textdata:
      return

    self.data = self.compute_representation(self.data)
    
    self.data = np.asarray(self.data, dtype=np.float32)
    self.labels = np.asarray(self.labels, dtype=np.int32)

    permutation = np.random.permutation(self.data.shape[0])
    self.data = self.data[permutation]
    self.labels = self.labels[permutation]

    print("Data size: " + str(self.data.shape))

  def save_additional_info(self):
    """ 
    Use Lucchese et al., 2011 dataset to complement columns in this dataset
    """

    original = aol_lucchese()
    original.load(permutation=False, textdata=True)

    csv_file = '/'.join(
      [download_datasets.DATA_DIR, 'aol', 'procheta_task_additional_info.csv'])
    with open(csv_file, mode='w') as data_file:
      writer = csv.writer(data_file, delimiter=',')
      row = ['cross-session task id', 'query', 'user id', 'time id', 'task id']
      writer.writerow(row)

      for i in range(len(self.data)):
        query = self.data[i]
        user_id, time_id, task_id, levenshtein = 0, 0, 0, 0xffffffff
        for j in range(len(original.data)):
          dist = distance.levenshtein(query, original.data[j])
          if dist < levenshtein:
            levenshtein = dist
            user_id = original.user_ids[j]
            time_id = original.time_ids[j]
            task_id = original.labels[j]
        row = [self.labels[i], query, user_id, time_id, task_id]
        writer.writerow(row)

class aol_lucchese(object):
  """
  Search session task dataset from (Lucchese et al., 2011)
  """

  def __init__(self, representation=representation().minhash):
    self.file = '/'.join(
      [download_datasets.DATA_DIR,'aol/aol-task-ground-truth','all-tasks.txt'])

    self.data       = []
    self.labels     = []
    self.user_ids   = []
    self.time_ids   = []
    self.same_tids  = []
    self.scaler     = None

    self.compute_representation = representation

  def load(self, permutation=True, textdata=False):
    """
    Load data into object variables according to the specified representation

    [1] http://miles.isti.cnr.it/~tolomei/?page_id=36
    """

    with open(self.file, mode='r') as data_file:
      reader = csv.reader(data_file, delimiter='\t')
      for row in reader:
        self.user_ids.append(int(row[0]))
        self.time_ids.append(int(row[1]))
        self.labels.append(int(row[2])) # task ID inside each time session
        # The third column has the original query ID
        self.data.append(row[4].strip()) 

    if textdata:
      return

    self.data = self.compute_representation(self.data)
    
    self.data = np.asarray(self.data, dtype=np.float32)
    self.labels = np.asarray(self.labels, dtype=np.int32)

    if permutation:
      permutation = np.random.permutation(self.data.shape[0])
      self.data = self.data[permutation]
      self.labels = self.labels[permutation]

    print("Data size: " + str(self.data.shape))

  def load_sequential_pair(self):
    """
    Load data in sequential pairs of queries, tagging with a one
    when there is a task change, zero otherwise
    """

    with open(self.file, mode='r') as data_file:
      reader = csv.reader(data_file, delimiter='\t')
      for row in reader:
        self.time_ids.append(int(row[1]))
        self.labels.append(int(row[2]))  # task ID inside each time session
        self.data.append(row[4].strip()) 

    pairs = []
    tags = []
    same_tids = []
    for i in range(len(self.data) - 1):
      pairs.append(self.data[i] + ' ' + self.data[i+1])

      if self.labels[i] == self.labels[i+1]:
        tags.append(0)
      else:
        tags.append(1)

      if self.time_ids[i] == self.time_ids[i+1]:
        same_tids.append(1)
      else:
        same_tids.append(0)

    self.data = self.compute_representation(pairs) 
    self.labels = tags
    self.same_tids = same_tids

    self.data = np.asarray(self.data, dtype=np.float32)
    self.labels = np.asarray(self.labels, dtype=np.int32)
    self.same_tids = np.asarray(self.same_tids, dtype=np.int32)

    permutation = np.random.permutation(self.labels.shape[0])
    self.data = self.data[permutation]
    self.labels = self.labels[permutation]
    self.same_tids = self.same_tids[permutation]

    print('Entry shape: ' + str(self.data[0].shape))
    print('Data shape:  ' + str(self.data.shape))

  def split_dataset(self):
    """
    Create training (70%) and testing (30%) sets for model learning
    """
    train_data, test_data, train_labels, test_labels = train_test_split(
      self.data, self.labels, test_size=0.3)
    return train_data, test_data, train_labels, test_labels

class aol_gayo(object):
  """
  Search session task dataset from (GayoAvello et al., 2006)
  """

  def __init__(self, representation=representation().minhash):
    self.file = '/'.join(
      [download_datasets.DATA_DIR,'aol', 'webis-smc-12.txt'])

    self.data       = []
    self.labels     = []
    self.user_ids   = []
    self.timestamps = []
    self.scaler     = None

    self.compute_representation = representation

  def load(self, permutation=True):
    """
    Load data into object variables according to the specified representation
    """

    with open(self.file, mode='r') as data_file:
      format = '%Y-%m-%d %H:%M:%S'
      now = datetime.now(timezone.utc)
      reader = csv.reader(data_file, delimiter='\t')
      for row in reader:
        if len(row) == 0 or row[0] == 'UserID' or row[0].startswith('---'):
          continue
        self.user_ids.append(int(row[0]))
        self.data.append(row[1].strip())
        self.timestamps.append(now.strptime(row[2], format))
        self.labels.append(int(row[5])) # Mission ID 
         
    self.data = self.compute_representation(self.data)
    
    self.data = np.asarray(self.data, dtype=np.float32)
    self.labels = np.asarray(self.labels, dtype=np.int32)

    if permutation:
      permutation = np.random.permutation(self.data.shape[0])
      self.data = self.data[permutation]
      self.labels = self.labels[permutation]

    print("Data size: " + str(self.data.shape))

  def load_sequential_pair(self):
    """
    Load data in sequential pairs of queries, tagging with a one
    when there is a task change, zero otherwise
    """

    with open(self.file, mode='r') as data_file:
      reader = csv.reader(data_file, delimiter='\t')
      for row in reader:
        if len(row) == 0 or row[0] == 'UserID' or row[0].startswith('---'):
          continue
        self.data.append(row[1].strip())
        self.labels.append(int(row[5])) # Mission ID 

    pairs = []
    tags = []
    for i in range(len(self.data) - 1):
      pairs.append(self.data[i] + ' ' + self.data[i+1])

      if self.labels[i] == self.labels[i+1]:
        tags.append(0)
      else:
        tags.append(1)

    self.data = self.compute_representation(pairs) 
    self.labels = tags

    self.data = np.asarray(self.data, dtype=np.float32)
    self.labels = np.asarray(self.labels, dtype=np.int32)
  
    permutation = np.random.permutation(self.labels.shape[0])
    self.data = self.data[permutation]
    self.labels = self.labels[permutation]
    
    print('Entry shape: ' + str(self.data[0].shape))
    print('Data shape:  ' + str(self.data.shape))

  def split_dataset(self):
    """
    Create training (70%) and testing (30%) sets for model learning
    """
    train_data, test_data, train_labels, test_labels = train_test_split(
      self.data, self.labels, test_size=0.3)
    return train_data, test_data, train_labels, test_labels

class reuters(object):
  """
  Reuters dataset with 4 major classes and 10k samples

  [1] https://github.com/XifengGuo/IDEC
  """
  def __init__(self):
    self.data       = []
    self.labels     = []

  def make_reuters_data(self, data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
      for line in fin.readlines():
        line = line.strip().split(' ')
        cat = line[0]
        did = int(line[1])
        if cat in cat_list:
          did_to_cat[did] = did_to_cat.get(did, []) + [cat]
      for did in list(did_to_cat.keys()):
        if len(did_to_cat[did]) > 1:
          del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
      with open(join(data_dir, dat)) as fin:
        for line in fin.readlines():
          if line.startswith('.I'):
            if 'did' in locals():
              assert doc != ''
              if did in did_to_cat:
                data.append(doc)
                target.append(cat_to_cid[did_to_cat[did][0]])
            did = int(line.strip().split(' ')[1])
            doc = ''
          elif line.startswith('.W'):
            assert doc == ''
          else:
            doc += line

    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000]
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print ('todense succeed')

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print ('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], x.size // x.shape[0]))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})

  def load(self, data_path='./datasets/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print ('making reuters idf features')
        self.make_reuters_data(data_path)
        print ('reutersidf saved to ' + data_path)
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy')).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], x.size // x.shape[0])).astype('float64')
    y = y.reshape((y.size,))

    self.data = x 
    self.labels = y

    print('Entry shape: ' + str(self.data[0].shape))
    print('Data shape:  ' + str(self.data.shape))

if __name__ == "__main__":
  print('datasets...')
  lucchese = aol_lucchese(representation=representation().fastText)
  lucchese.load_sequential_pair()
  lucchese.split_dataset()
