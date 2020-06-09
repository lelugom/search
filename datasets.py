"""
Load data from local folder and encode information according to the selected
representation
"""

import download_datasets

import io, csv, gc, os, sys, distance, json
import re, urllib.request, urllib.parse
from datetime import timezone, datetime

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np 
import tf_sentencepiece

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.test.utils import get_tmpfile
from gensim.models.fasttext import load_facebook_vectors
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from scipy.spatial.distance import cosine

# Print the whole arrays and set random seed
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(43)

class representation(object):
  """
  Implement different representation approaches to encode text data
  """
  def __init__(self, lang='en', width=0):
    self.vectors = None
    self.session = None
    self.lang    = lang
    self.width   = width

    self.muse_cache_file = download_datasets.DATA_DIR + '/muse/cache_muse.csv'
    self.muse_cache_sep  = '\t|\t|\t|\t'
    self.muse_cache = {}

  def clear_cache(self):
    """
    Remove dictionaries for representation caches
    """
    self.muse_cache = {}
    print('\tGC collect representation' + str(gc.collect()))

  def store_muse_cache(self):
    """
    Store cache into local disk using a CSV file
    """
    dir_path = os.path.dirname(self.muse_cache_file)
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

    with open(self.muse_cache_file, mode='w') as data_file:
      for query in self.muse_cache:
        embedding = [str(number) for number in self.muse_cache[query]]
        embedding = ','.join(embedding)
        data_file.write(self.muse_cache_sep.join([query, embedding]) + '\n')
        
  def load_muse_cache(self):
    """
    Load CSV file from local disk and store query MUSE embeddings into cache 
    local variable
    """
    if not os.path.exists(self.muse_cache_file):
      return

    print('\tLoading MUSE cache ' + self.muse_cache_file)
    with open(self.muse_cache_file, mode='r') as data_file:
      for line in data_file:
        line = line.strip()
        row = line.split(self.muse_cache_sep)
        query = row[0]
        embedding = [float(number) for number in row[1].split(',')]
        self.muse_cache[query] = embedding

  def get_muse_vectors(self, queries):
    """
    Load query embeddings from the MUSE cache. If a query is not stored there,
    load the MUSE model and update the cache
    """
    embeddings = []
    update_cache = False

    for query in queries:
      embedding = self.muse_cache.get(query, None)
      if embedding == None:
        update_cache = True
        break
      embeddings.append(embedding)

    if update_cache == False:
      return embeddings

    muse = self.vectors(queries)
    embeddings = self.session.run(muse)
    for i in range(len(queries)):
      self.muse_cache[queries[i]] = embeddings[i]
    self.store_muse_cache()
    del muse
    return embeddings

  def universal_sentence_encoder(self, text_data):
    """
    Pretrained universal sentence encoder 

    [1] https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html
    """
    embeddings = []
    config = tf.ConfigProto()
    
    if self.vectors == None:
      self.session = tf.Session(config=config)
      self.vectors = hub.Module(download_datasets.M_UNIVERSAL_SENTENCE_ENCODER)
      self.session.run(tf.global_variables_initializer())
      self.session.run(tf.tables_initializer())
      self.load_muse_cache()

    if isinstance(text_data[0], list):
      for queries in text_data:
        muse_embeddings = self.get_muse_vectors(queries)
        embeddings.append(muse_embeddings)
    else:
      step = 10000
      for index in range(0, len(text_data), step):
        muse_embeddings = self.get_muse_vectors(text_data[index : index + step])
        embeddings.extend(muse_embeddings)

    return embeddings
    
  def glove(self, text_data):
    """
    Pretrained word embeddings from GloVe

    [1] https://nlp.stanford.edu/projects/glove/
    [2] https://radimrehurek.com/gensim/scripts/glove2word2vec.html
    """

    if self.vectors == None:
      glove2word2vec('datasets/glove/glove.42B.300d.txt', 
        'datasets/glove/glove.42B.300d_word2vec.txt')
      self.vectors = KeyedVectors.load_word2vec_format(
        'datasets/glove/glove.42B.300d_word2vec.txt')
    return self.embeddings(text_data)

  def embeddings(self, text_data):
    """
    If self.width is zero, determine if samples are sentences or arrays of 
    sentences to compute embeddings; then, return one average vector for 
    sentences and an array of average vectors for arrays of sentences. If 
    self.width is greater than zero, use that width as the number of word 
    embeddings for a sentece.
    """

    if self.width > 0:
      return self.compute_embeddings(text_data)
    else:
      if isinstance(text_data[0], list):
        return self.compute_array_avg_embeddings(text_data)
      else:
        return self.compute_avg_embeddings(text_data)

  def compute_embeddings(self, text_data):
    """
    Calculate word embeddings for every sentence in text_data array. Use the 
    model stored in vectors to retrieve pretrained word embeddings.
    """

    embedding_length = len(self.vectors['tree'])
    query_length = self.width

    embeddings = []
    for query in text_data:
      not_in_vocabulary = [0.0 for i in range(embedding_length)]
      embedding = np.zeros((query_length, embedding_length))

      for index, word in enumerate(query.split()):
        if index == query_length:
          break

        try:
          vector = self.vectors[word]
        except:
          vector = not_in_vocabulary
        embedding[index] = np.asarray(vector, dtype=np.float32)
      embeddings.append(embedding)

    return np.asarray(embeddings, dtype=np.float32)

  def compute_avg_embeddings(self, text_data):
    """
    Calculate word embeddings for every sentence in text_data array. Use the 
    model stored in vectors to retrieve pretrained word embeddings and compute
    the average of word vectors
    """

    embedding_length = len(self.vectors['tree'])
    embeddings = []

    for query in text_data:
      not_in_vocabulary = [0.0 for i in range(embedding_length)]
      embedding = []
      
      for _, word in enumerate(query.split()):
        try:
          vector = self.vectors[word]
        except:
          vector = not_in_vocabulary
        embedding.append(vector)

      if len(embedding) == 0:
        embedding.append(not_in_vocabulary)

      embedding = np.asarray(embedding, dtype=np.float32)
      embeddings.append(np.mean(embedding, axis=0))

    return embeddings

  def compute_array_avg_embeddings(self, text_data):
    """
    Calculate word embeddings for every array of sentences in text_data array. 
    Use the model stored in vectors to retrieve pretrained word embeddings 
    and compute the average of word vectors
    """

    embedding_length = len(self.vectors['tree'])
    embeddings = []

    for queries in text_data: 
      embedding_array = []
      for query in queries:
        not_in_vocabulary = [0.0 for i in range(embedding_length)]
        embedding = []
        
        for _, word in enumerate(query.split()):
          try:
            vector = self.vectors[word]
          except:
            vector = not_in_vocabulary
          embedding.append(vector)

        if len(embedding) == 0:
          embedding.append(not_in_vocabulary)

        embedding = np.asarray(np.mean(embedding, axis=0), dtype=np.float32)
        embedding_array.append(embedding)
      embeddings.append(embedding_array)

    return embeddings

class clueweb(object):
  """
  Retrieval of search results from ClueWeb 

  [1] https://github.com/procheta/AOLTaskExtraction
  """
  def __init__(self, representation=representation().glove):
    self.BASE_URL = ''
    self.CACHE_FILE = download_datasets.DATA_DIR + '/clueweb/cache.csv'
    self.CACHE_SEP  = '\t|\t|\t|\t'
    self.cache = {}
    self.documents_ids = {}
    self.representation=representation

    self.load_cache()

  def load_cache(self):
    """
    Load CSV file from local disk and store query JSON results into cache 
    local variable
    """
    if not os.path.exists(self.CACHE_FILE):
      return

    print('\tLoading ClueWeb cache ' + self.CACHE_FILE)
    with open(self.CACHE_FILE, mode='r') as data_file:
      for line in data_file:
        line = line.strip()
        row = line.split(self.CACHE_SEP)
        self.cache[row[0]] = row[1]
      
  def save_cache(self):
    """
    Store cache into local disk using a CSV file
    """
    dir_path = os.path.dirname(self.CACHE_FILE)
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

    with open(self.CACHE_FILE, mode='w') as data_file:
      for query in self.cache:
        data_file.write(self.CACHE_SEP.join([query, self.cache[query]]) + '\n')

  def retrieve(self, query):
    """
    Connect to the ClueWeb endpoint and retrieve JSON results. Update cache file
    to avoid unnecesary connections
    """
    results = self.cache.get(query, None)
    if results != None:
      return results

    url = self.BASE_URL + urllib.parse.quote_plus(query)
    print('Retrieving %s ' % url)
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(request)
    
    results = html.read().decode("utf-8")
    results = re.sub(r'(\\n|\\t)+', ' ', results)
    results = re.sub(r'\s+', ' ', results)
    results = re.sub(r'\<.*?B\>', '', results)
    
    self.cache[query] = results
    self.save_cache()
    return results
  
  def decode_json(self, json_message=None):
    """
    [1] https://stackoverflow.com/questions/1505454/python-json-loads-chokes-on-escapes
    """
    result = None
    try:        
      result = json.loads(json_message)
    except Exception as e:      
      # Find the offending character index:
      idx_to_replace = int(str(e).split(' ')[-1].replace(')', ''))    
      # Remove the offending character:
      json_message = list(json_message)
      json_message[idx_to_replace] = ' '
      new_message = ''.join(json_message)     
      return self.decode_json(json_message=new_message)
    return result

  def retrieve_document_ids(self, query):
    """
    Return an array of document id strings retrieved when submitting the query
    """
    ids = self.documents_ids.get(query, None)
    if ids != None:
      return ids
 
    ids = []
    results = self.retrieve(query)
    if results.startswith('Nothing found!!'):
      self.documents_ids[query] = ids
      return ids

    decoded = self.decode_json(results)
    for result in decoded:
      try:
        id = result[0]['id']
        ids.append(id)
      except:
        print('No id found')

    self.documents_ids[query] = ids
    return ids
  
  def semantic_similarity_ids(self, q0, q1):
    """
    Use the ids from the first documents retrieved from ClueWeb to 
    calculate the semantic similarity
    """
    if q0 == q1:
      return 1.0

    q0_ids = self.retrieve_document_ids(q0)
    q1_ids = self.retrieve_document_ids(q1)

    initial_size = len(q0_ids)
    for id in q1_ids:
      if id in q0_ids:
        q0_ids.remove(id)

    intersec_size = initial_size - len(q0_ids)
    if len(q0_ids) != 0 and len(q1_ids) != 0:
      if q0 != q1:
        return float(intersec_size) / (
          float(initial_size + len(q1_ids) - intersec_size))
      else:
        return 1
    else:
     return 0

class csv_base_dataset():
  """
  Base class for a dataset load from an CSV stored in disk
  """
  def __init__(self, representation=representation().glove):
    self.data    = []
    self.labels  = []
    self.scaler  = None
    self.file    = ''
    self.k       = 10
    self.splits  = None
    
    self.delimiter     = ','
    self.file_encoding = 'UTF-8'

    self.compute_representation = representation

  def load(self, textdata=False, prefix_title=None, data_idx=0, label_idx=1):
    """
    Load data into object variables according to the specified representation.
    If textdata is True, do not compute any representation for the queries
    """
    label_dict = {}

    with open(self.file, mode='r', encoding=self.file_encoding) as data_file:
      reader = csv.reader(data_file, delimiter=self.delimiter)
      try:
        for row in reader:
          if prefix_title != None and row[0].startswith(prefix_title):
            continue

          text_label = row[label_idx]
          if label_dict.get(text_label, None) == None:
            label_dict[text_label] = len(label_dict)
          self.labels.append(label_dict[text_label])

          query = row[data_idx] 
          query = query.strip()
          if self.file_encoding == 'gb18030': 
            query = query.replace('[', '').replace(']', '')
          self.data.append(query)
      except Exception as e:
        print('Exception while loading dataset')
        print('Loaded %d queries and %d labels \n%s'%(
          len(self.data), len(self.labels), e))

    self.labels = np.asarray(self.labels, dtype=np.int32)
    if textdata == False:
      self.data = self.compute_representation(self.data)
      self.data = np.asarray(self.data, dtype=np.float32)
      print("Data size: " + str(self.data.shape))

  def split_dataset(self, test_size=0.3):
    """
    Create training (70%) and testing (30%) sets for model learning
    """
    train_data, test_data, train_labels, test_labels = train_test_split(
      self.data, self.labels, test_size=test_size)
    return train_data, test_data, train_labels, test_labels

  def kfold(self, test_size=0.1, k=10):
    """
    Create the folds for cross validation. Use shuffle split to create folds
    randomly
    """
    self.k = k
    kfold = ShuffleSplit(n_splits=self.k, test_size=test_size)
    self.splits = kfold.split(X=self.data, y=self.labels)

  def next_fold(self):
    """
    Get next fold indices. Update train and test sets
    """
    train_indices, test_indices = next(self.splits)
    self.train_data    = self.data[train_indices]
    self.train_labels  = self.labels[train_indices]
    self.test_data     = self.data[test_indices]
    self.test_labels   = self.labels[test_indices]
    
class volske_aol(csv_base_dataset):
  """
  Volske et al., 2019 dataset based on AOL
  """
  def __init__(self, representation=representation().glove):
    super().__init__(representation)
    self.file = '/'.join(
      [download_datasets.DATA_DIR, 'volske', 'd1_session.csv'])

  def load(self, textdata=False):
    super().load(
      textdata=textdata, prefix_title='Query', data_idx=0, label_idx=2)

class volske_trek(csv_base_dataset):
  """
  Volske et al., 2019 dataset based on Trec
  """
  def __init__(self, representation=representation().glove):
    super().__init__(representation)
    self.file = '/'.join(
      [download_datasets.DATA_DIR, 'volske', 'd2_trec.csv'])

  def load(self, textdata=False):
    super().load(
      textdata=textdata, prefix_title='Query', data_idx=0, label_idx=2)

class volske_wikihow(csv_base_dataset):
  """
  Volske et al., 2019 dataset based on WikiHow
  """
  def __init__(self, representation=representation().glove):
    super().__init__(representation)
    self.file = '/'.join(
      [download_datasets.DATA_DIR, 'volske', 'd3_wikihow.csv'])

  def load(self, textdata=False):
    super().load(
      textdata=textdata, prefix_title='Query', data_idx=0, label_idx=2)

class sen_aol(csv_base_dataset):
  """
  Search session task dataset from (Sen et al., 2018)
  """
  def __init__(self, representation=representation().glove):
    super().__init__(representation)
    self.file = '/'.join(
      [download_datasets.DATA_DIR, 'aol', 'procheta_task.csv'])

  def load(self, textdata=False):
    super().load(
      textdata=textdata, prefix_title=None, data_idx=0, label_idx=1)

  def save_additional_info(self):
    """ 
    Use Lucchese et al., 2011 dataset to complement columns in this dataset
    """

    original = lucchese_aol()
    original.load(textdata=True)

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

class lucchese_aol(object):
  """
  Search session task dataset from (Lucchese et al., 2011)
  """

  def __init__(self, 
    representation=representation().glove):

    self.file = '/'.join(
      [download_datasets.DATA_DIR,'aol/aol-task-ground-truth','all-tasks.txt'])
    self.file_additional_info = '/'.join(
      [download_datasets.DATA_DIR, 
      'aol/aol-task-ground-truth', 'all-tasks_additional_info.txt'])

    self.data       = []
    self.labels     = []
    self.user_ids   = []
    self.time_ids   = []
    self.query_ids  = []
    self.cross_session_labels = []
    self.timestamps = []
    self.same_tids  = []
    self.scaler     = None

    self.max_gap    = 0
    
    self.compute_representation = representation
    self.compute_semantic = None

  def load(self, textdata=False, semantic=False):
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
        self.query_ids.append(int(row[3])) # original query ID
        self.data.append(row[4].strip()) 

    if textdata:
      return

    lexic = self.compute_representation(self.data)
    lexic = np.asarray(lexic, dtype=np.float32)

    if semantic:
      semantic = self.compute_semantic(self.data)
      semantic = np.asarray(semantic, dtype=np.float32)
      self.data = np.concatenate((lexic, semantic), axis=1)
    else:
      self.data = lexic

    self.labels = np.asarray(self.labels, dtype=np.int32)

    print("Data size: " + str(self.data.shape))

  def load_additional_info(self, textdata=False, semantic=False):
    """
    Load CSV file with additional infor for the dataset
    """

    self.read_csv_additional_info()

    if textdata:
      return

    lexic = self.compute_representation(self.data)
    lexic = np.asarray(lexic, dtype=np.float32)

    if semantic:
      semantic = self.compute_semantic(self.data)
      semantic = np.asarray(semantic, dtype=np.float32)
      self.data = np.concatenate((lexic, semantic), axis=1)
    else:
      self.data = lexic

    self.labels = np.asarray(self.labels, dtype=np.int32)
    self.cross_session_labels = np.asarray(
      self.cross_session_labels, dtype=np.int32)

    print("Data size: " + str(self.data.shape))

  def read_csv_additional_info(self):
    """
    Read and parse CSV file with additional info for the dataset
    """

    if not os.path.exists(self.file_additional_info):
      self.load(textdata=True)
      print('\tCreating log with additional info ...')
      self.save_additional_info()

    with open(self.file_additional_info, mode='r') as data_file:
      time_format = '%Y-%m-%d %H:%M:%S'
      now = datetime.now(timezone.utc)
      reader = csv.reader(data_file, delimiter='\t')
      for row in reader:
        self.user_ids.append(int(row[0]))
        self.time_ids.append(int(row[1]))
        self.labels.append(int(row[2])) # task ID inside each time session
        self.query_ids.append(int(row[3])) # original query ID
        self.data.append(row[4].strip()) # query
        self.cross_session_labels.append(int(row[5]))
        self.timestamps.append(now.strptime(row[6], time_format))

  def load_sequential_pair(self):
    """
    Load data in sequential pairs of queries, tagging with a one
    when there is a task change, zero otherwise. Use cross-session tasks 
    in Sen et al., 2018 instead of intra session labels
    """

    self.read_csv_additional_info()
    self.labels = self.cross_session_labels

    pairs, tags, gaps = [], [], []
    for i in range(len(self.data) - 1):
      tdelta = self.timestamps[i] - self.timestamps[i+1]
      time_gap = float(abs(tdelta.total_seconds()))
      gaps.append(time_gap)
      pairs.append([self.data[i], self.data[i+1], time_gap, '', ''])
      if self.labels[i] == self.labels[i+1]:
        tags.append(0)
      else:
        tags.append(1)

    self.data = pairs 
    self.labels = tags
    
    self.max_gap = max(gaps)

    print('Entry shape: ' + str(np.asarray(self.data[0]).shape))
    print('Data shape:  ' + str(np.asarray(self.data).shape))

  def load_sequential_queries(self, m=0, n=1):
    """
    Load a sequential group of m + n + 1 queries, tagging with a one
    when there is a task change, zero otherwise. Use cross-session tasks 
    in Sen et al., 2018 instead of intra session labels. Include time gap information
    """

    assert(n > 0)
    self.read_csv_additional_info()
    self.labels = self.cross_session_labels

    groups, tags, gaps = [], [], [] 
    for i in range(len(self.data) - 1):
      group = []
      for j in range(i - m, i):
        if j >= 0:
          group.append(self.data[j])
        else:
          group.append('')

      for j in range(i, i + n + 1):
        if j < len(self.data):
          group.append(self.data[j])
        else:
          group.append('')
          
      groups.append(group)

      tdelta = self.timestamps[i] - self.timestamps[i+1]
      gaps.append(float(tdelta.total_seconds()))
      if self.labels[i] == self.labels[i+1]:
        tags.append(0)
      else:
        tags.append(1)
    
    self.data = self.compute_representation(groups) 
    self.labels = tags
    self.data = np.asarray(self.data, dtype=np.float32)
    self.labels = np.asarray(self.labels, dtype=np.int32)

    gaps = np.asarray(gaps, dtype=np.float32)
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)
    gaps -= mean_gap
    gaps /= std_gap

    temp = []
    data_len = len(self.data[0, 0])
    for i in range(len(self.data)):
      time_gap = np.full((1, data_len), gaps[i])
      temp.append(np.append(self.data[i], time_gap, axis=0))
    self.data = np.asarray(temp, dtype=np.float32)

    print('Entry shape: ' + str(self.data[0].shape))
    print('Data shape:  ' + str(self.data.shape))

  def split_dataset(self, test_size=0.3):
    """
    Create training (70%) and testing (30%) sets for model learning
    """
    train_data, test_data, train_labels, test_labels = train_test_split(
      self.data, self.labels, test_size=test_size)
    return train_data, test_data, train_labels, test_labels

  def kfold(self, test_size=0.1, k=10):
    """
    Create the folds for cross validation. Use shuffle split to create folds
    randomly
    """
    self.k = k
    kfold = ShuffleSplit(n_splits=self.k, test_size=test_size)
    self.splits = kfold.split(X=self.data, y=self.labels)

  def next_fold(self):
    """
    Get next fold indices. Update train and test sets
    """
    train_indices, test_indices = next(self.splits)
    self.train_data    = self.data[train_indices]
    self.train_labels  = self.labels[train_indices]
    self.test_data     = self.data[test_indices]
    self.test_labels   = self.labels[test_indices]
    
  def save_additional_info(self):
    """ 
    Use Sen et al., 2018 and original AOL dataset to create two additional
    columns in this dataset: cross session tasks identifiers and timestamps. 
    The Levenshtein distance is used to match the queries between the datasets.
    """

    sen = sen_aol()
    sen.load(textdata=True)

    uids = set(self.user_ids)
    anonids, queries, querytimes = [], [], []
    csv_file = '/'.join(
      [download_datasets.DATA_DIR, 
      'aol/AOL-user-ct-collection', 'user-ct-test-collection.txt'])
    with open(csv_file, mode='r') as data_file:
      reader = csv.reader(data_file, delimiter='\t')
      for row in reader:
        if row[0] == 'AnonID':
          continue
        try:
          id = int(row[0])
        except:
          id = 0
          print('Unable to parse AnonID ' + row[0])
        if id in uids:
          anonids.append(id)
          queries.append(row[1])
          querytimes.append(row[2])

    csv_file = self.file_additional_info 
    with open(csv_file, mode='w') as data_file:
      writer = csv.writer(data_file, delimiter="\t")

      for i in range(len(self.data)):
        query = self.data[i]
        cross_session_task_id, levenshtein = 0, 0xffffffff
        for j in range(len(sen.data)):
          dist = distance.levenshtein(query, sen.data[j])
          if dist < levenshtein:
            levenshtein = dist
            cross_session_task_id = sen.labels[j]

        timestamp, levenshtein = '', 0xffffffff
        for j in range(len(queries)):
          if anonids[j] != self.user_ids[i]:
            continue
          dist = distance.levenshtein(query, queries[j])
          if dist < levenshtein:
            levenshtein = dist
            timestamp = querytimes[j]
        row = [self.user_ids[i], self.time_ids[i], self.labels[i], 
          self.query_ids[i], query, cross_session_task_id, timestamp]
        writer.writerow(row)

class hagen_aol(object):
  """
  Search session task dataset from (Hagen et al., 2013)
  """

  def __init__(self, 
    representation=representation().glove):
    self.file = '/'.join(
      [download_datasets.DATA_DIR,'aol', 'webis-smc-12.txt'])

    self.data       = []
    self.labels     = []
    self.user_ids   = []
    self.timestamps = []
    self.urls       = []
    self.scaler     = None
    self.time_ids   = []

    self.user_sessions   = []
    self.n_user_sessions = 0
    self.user_session_data   = []
    self.user_session_labels = []

    self.max_gap    = 0

    self.compute_representation = representation
    self.compute_semantic = None

    self.k = 10
    self.splits = None

  def read_csv(self):
    """
    Read and parse CSV file for the dataset
    """

    with open(self.file, mode='r') as data_file:
      time_format = '%Y-%m-%d %H:%M:%S'
      now = datetime.now(timezone.utc)
      reader = csv.reader(data_file, delimiter='\t')
      for row in reader:
        if len(row) == 0 or row[0] == 'UserID' or row[0].startswith('---'):
          continue
        self.user_ids.append(int(row[0]))
        self.data.append(row[1].strip())
        self.timestamps.append(now.strptime(row[2], time_format))
        self.urls.append(row[4].strip())
        self.labels.append(int(row[5])) # Mission ID 

  def load(self, semantic=False):
    """
    Load data into object variables according to the specified representation
    """

    self.read_csv()
         
    lexic = self.compute_representation(self.data)
    lexic = np.asarray(lexic, dtype=np.float32)

    if semantic:
      semantic = self.compute_semantic(self.data)
      semantic = np.asarray(semantic, dtype=np.float32)
      self.data = np.concatenate((lexic, semantic), axis=1)
    else:
      self.data = lexic

    self.labels = np.asarray(self.labels, dtype=np.int32)

    print("Data size: " + str(self.data.shape))

  def compute_time_ids(self):
    """
    Use query timestamp and user IDs in order to compute time ids
    """
    time_id = 1
    user_id = 0
    self.time_ids.append(time_id)

    for i in range(1, len(self.timestamps)):
      if user_id != self.user_ids[i]:
        time_id = 1
        user_id = self.user_ids[i]
      else:
        tdelta = self.timestamps[i] - self.timestamps[i - 1]        
        gap = float(abs(tdelta.total_seconds()))
        if gap > 60 * 26:
          time_id += 1
      
      self.time_ids.append(time_id)

  def load_sequential_pair(self):
    """
    Load data in sequential pairs of queries, tagging with a one
    when there is a task change, zero otherwise. Include time gap in the vectors
    """
    self.read_csv()

    pairs, tags, gaps = [], [], []
    for i in range(len(self.data) - 1):
      tdelta = self.timestamps[i] - self.timestamps[i+1]
      time_gap = float(abs(tdelta.total_seconds()))
      gaps.append(time_gap)
      pairs.append(
        [self.data[i], self.data[i+1], time_gap, self.urls[i], self.urls[i+1]])
      if self.labels[i] == self.labels[i+1]:
        tags.append(0)
      else:
        tags.append(1)

    self.data = pairs 
    self.labels = tags
    
    self.max_gap = max(gaps)

    print('Entry shape: ' + str(np.asarray(self.data[0]).shape))
    print('Data shape:  ' + str(np.asarray(self.data).shape))

  def load_sequential_queries(self, m=0, n=1):
    """
    Load a sequential group of m + n + 1 queries, tagging with a one
    when there is a task change, zero otherwise. Include time gap information
    """

    assert(n > 0)
    self.read_csv()

    groups, tags, gaps = [], [], [] 
    for i in range(len(self.data) - 1):
      group = []
      for j in range(i - m, i):
        if j >= 0:
          group.append(self.data[j])
        else:
          group.append('')

      for j in range(i, i + n + 1):
        if j < len(self.data):
          group.append(self.data[j])
        else:
          group.append('')
          
      groups.append(group)

      tdelta = self.timestamps[i] - self.timestamps[i+1]
      gaps.append(float(tdelta.total_seconds()))
      if self.labels[i] == self.labels[i+1]:
        tags.append(0)
      else:
        tags.append(1)
    
    self.data = self.compute_representation(groups) 
    self.labels = tags
    self.data = np.asarray(self.data, dtype=np.float32)
    self.labels = np.asarray(self.labels, dtype=np.int32)

    gaps = np.asarray(gaps, dtype=np.float32)
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)
    gaps -= mean_gap
    gaps /= std_gap

    temp = []
    data_len = len(self.data[0, 0])
    for i in range(len(self.data)):
      time_gap = np.full((1, data_len), gaps[i])
      temp.append(np.append(self.data[i], time_gap, axis=0))
    self.data = np.asarray(temp, dtype=np.float32)

    print('Entry shape: ' + str(self.data[0].shape))
    print('Data shape:  ' + str(self.data.shape))

  def split_user_sessions(self):
    """
    Find user sessions to split dataset per user ID
    """
    ids = set(self.user_ids)
    self.n_user_sessions = len(ids)
    self.user_sessions = []

    for id in ids:
      self.user_sessions.append(
        [i for i, value in enumerate(self.user_ids) if value == id])
    self.user_sessions = iter(self.user_sessions)

  def next_user_session(self):
    """
    Get next session indices. Update data set
    """
    indices = next(self.user_sessions)
    self.user_session_data   = self.data[indices]
    self.user_session_labels = self.labels[indices]

  def split_dataset(self, test_size=0.3):
    """
    Create training (70%) and testing (30%) sets for model learning
    """
    train_data, test_data, train_labels, test_labels = train_test_split(
      self.data, self.labels, test_size=test_size)
    return train_data, test_data, train_labels, test_labels

  def kfold(self, test_size=0.1, k=10):
    """
    Create the folds for cross validation. Use shuffle split to create folds
    randomly
    """
    self.k = k
    kfold = ShuffleSplit(n_splits=self.k, test_size=test_size)
    self.splits = kfold.split(X=self.data, y=self.labels)

  def next_fold(self):
    """
    Get next fold indices. Update train and test sets
    """
    train_indices, test_indices = next(self.splits)
    self.train_data    = self.data[train_indices]
    self.train_labels  = self.labels[train_indices]
    self.test_data     = self.data[test_indices]
    self.test_labels   = self.labels[test_indices]
