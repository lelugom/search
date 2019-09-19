"""
Load data from local folder and encode information according to the selected
representation
"""

import download_datasets

import io, csv, os, sys, distance, json
import re, urllib.request, urllib.parse
import numpy as np 

from datetime import timezone, datetime

import tensorflow_hub as hub
import tensorflow as tf

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
  def __init__(self):
    self.vectors = None

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

  def universal_sentence_encoder(self, text_data):
    """
    Pretrained universal sentence encoder 

    [1] https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html
    """
    embeddings = []

    if self.vectors == None:
      self.vectors = hub.Module(
        "https://tfhub.dev/google/universal-sentence-encoder-large/3")

    config = tf.ConfigProto()
    config.graph_options.rewrite_options.shape_optimization = 2

    with tf.Session(config=config) as session:
      session.run(tf.global_variables_initializer())
      session.run(tf.tables_initializer())

      if isinstance(text_data[0], list):
        for queries in text_data:
          use = self.vectors(queries)
          embeddings.append(session.run(use))
      else:
        use = self.vectors(text_data)
        embeddings = session.run(use)

    return embeddings
      
  def fastText(self, text_data):
    """
    Pretrained word embeddings from fastText

    [1] https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
    [2] https://radimrehurek.com/gensim/models/fasttext.html
    """
    
    if self.vectors == None:
      model = load_facebook_vectors('datasets/fasttext/cc.en.300.bin')
      self.vectors = model
    return self.avg_embeddings(text_data)
      
  def word2vec(self, text_data):
    """
    Pretrained word embeddings from word2vec

    [1] https://code.google.com/archive/p/word2vec/
    """
    if self.vectors == None:
      self.vectors = KeyedVectors.load_word2vec_format(
        'datasets/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    return self.avg_embeddings(text_data)
    
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
    return self.avg_embeddings(text_data)

  def avg_embeddings(self, text_data):
    """
    Determine if samples are sentences or arrays of sentences to compute 
    embeddings. Return one average vector for sentences and an array of average
    vectors for arrays of sentences
    """

    if isinstance(text_data[0], list):
      return self.compute_array_avg_embeddings(text_data)
    else:
      return self.compute_avg_embeddings(text_data)

  def compute_embeddings(self, text_data):
    """
    Calculate embeddings for every sentence in text_data array. Use the 
    model stored in vectors to retrieve pretrained word embeddings
    """

    embedding_length = len(self.vectors['tree'])
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
          vector = self.vectors[word]
        except:
          vector = not_in_vocabulary
          #print('\t  ' + word + ' not in vocabulary')
        embedding[index] = np.asarray(vector, dtype=np.float32)

      embeddings.append(embedding)

    return embeddings

  def compute_avg_embeddings(self, text_data):
    """
    Calculate embeddings for every sentence in text_data array. Use the 
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
          #print('\t  ' + word + ' not in vocabulary')
        embedding.append(vector)

      if len(embedding) == 0:
        embedding.append(not_in_vocabulary)

      embedding = np.asarray(embedding, dtype=np.float32)
      embeddings.append(np.mean(embedding, axis=0))

    return embeddings

  def compute_array_avg_embeddings(self, text_data):
    """
    Calculate embeddings for every array of sentences in text_data array. 
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
            #print('\t  ' + word + ' not in vocabulary')
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
  """
  def __init__(self, representation=representation().word2vec):
    self.BASE_URL = 'http://clueweb.adaptcentre.ie/WebSearcher/search?query='
    self.CACHE_FILE = download_datasets.DATA_DIR + '/clueweb/cache.csv'
    self.CACHE_SEP  = "\t|\t|\t|\t"
    self.cache = {}
    self.representation=representation

    self.load_cache()

  def load_cache(self):
    """
    Load CSV file from local disk and store query JSON results into cache 
    local variable
    """
    if not os.path.exists(self.CACHE_FILE):
      return

    with open(self.CACHE_FILE, mode='r') as data_file:
      for line in data_file:
        line = line.strip()
        row = line.split(self.CACHE_SEP)
        self.cache[row[0]] = row[1]
      
  def save_cache(self):
    """
    Store cache into local dis using a CSV file
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
    url = urllib.request.urlopen(url)
    
    results = url.read().decode("utf-8")
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

  def retrieve_array_avg_embeddings(self, query):
    """
    Return an average vector for every document snippets retrieved when 
    executing the query 
    """
    snippets = []
    results = self.retrieve(query)
    if results.startswith('Nothing found!!'):
      # No results. Use query text for semantic context
      results = '[[{"title":"' + query + '"}]]' 
    decoded = self.decode_json(results)

    for result in decoded:
      try:
        snippet = result[0]['snippet']
      except:
        # No snippet. Use document title instead
        snippet = result[0]['title']       
      snippets.append(snippet)

    embbeddings = self.representation(snippets)
    return embbeddings

  def retrieve_avg_embeddings(self, query):
    """ 
    Return an average vector for all the document snippets retrieved 
    when executing the query
    """

    embeddings = self.retrieve_array_avg_embeddings(query)
    return np.mean(embeddings, axis=0)

  def compute_avg_embeddings(self, text_data):
    """
    Calculate semantic average embeddings for every query in text_data array
    """

    embeddings = []
    for query in text_data:
      embeddings.append(self.retrieve_avg_embeddings(query))
    return embeddings

  def semantic_similarity(self, q0, q1):
    """
    Compute the average of cosine similarities between the average vectors of
    the first 1000 snippets in ClueWeb
    """
    similarities = []
    q0_embeddings = self.retrieve_array_avg_embeddings(q0)
    q1_embeddings = self.retrieve_array_avg_embeddings(q1)
    
    num_embeddings = min(len(q0_embeddings), len(q1_embeddings))
    for i in range(num_embeddings):
      sim = 1 - cosine(q0_embeddings[i], q1_embeddings[i])
      if not np.isnan(sim):
        similarities.append(sim)
  
    return np.average(similarities)
    
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

    print(self.data.shape)

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
    print(self.data.shape)

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
    Load data into object variables according to the specified representation. If textdata is True, do not compute any representation for the queries
    """

    with open(self.file, mode='r') as data_file:
      reader = csv.reader(data_file, delimiter=',')
      for row in reader:
        self.data.append(row[0].strip())
        self.labels.append(int(row[1]))
    self.labels = np.asarray(self.labels, dtype=np.int32)

    if textdata:
      return

    self.data = self.compute_representation(self.data)
    self.data = np.asarray(self.data, dtype=np.float32)

    if np.min(self.labels) != 0:
      print('Adjusting labels to start from 0')
      self.labels -= 1
      assert np.min(self.labels) == 0
      
    print("Data size: " + str(self.data.shape))

  def save_additional_info(self):
    """ 
    Use Lucchese et al., 2011 dataset to complement columns in this dataset
    """

    original = aol_lucchese()
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

class aol_lucchese(object):
  """
  Search session task dataset from (Lucchese et al., 2011)
  """

  def __init__(self, 
    representation=representation().word2vec):

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

    self.compute_representation = representation
    self.compute_semantic = clueweb(
      representation=representation).compute_avg_embeddings

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

    assert os.path.exists(self.file_additional_info)

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
    in Sen et al., 2018 instead of intra session labels. Include time gap 
    in the vectors
    """

    self.read_csv_additional_info()
    self.labels = self.cross_session_labels

    pairs, tags, gaps = [], [], [] 
    for i in range(len(self.data) - 1):
      pairs.append([self.data[i], self.data[i+1]])
      tdelta = self.timestamps[i] - self.timestamps[i+1]
      gaps.append(float(abs(tdelta.total_seconds())))
      if self.labels[i] == self.labels[i+1]:
        tags.append(0)
      else:
        tags.append(1)

    self.data = self.compute_representation(pairs) 
    self.labels = tags
    self.data = np.asarray(self.data, dtype=np.float32)
    self.labels = np.asarray(self.labels, dtype=np.int32)

    gaps = np.asarray(gaps, dtype=np.float32)
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)
    gaps -= mean_gap
    gaps /= std_gap

    data_len = len(self.data[0, 0])
    temp = []
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

    sen = aol_procheta()
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

class aol_gayo(object):
  """
  Search session task dataset from (GayoAvello et al., 2006)
  """

  def __init__(self, 
    representation=representation().minhash, 
    semantic=clueweb().compute_avg_embeddings):
    self.file = '/'.join(
      [download_datasets.DATA_DIR,'aol', 'webis-smc-12.txt'])

    self.data       = []
    self.labels     = []
    self.user_ids   = []
    self.timestamps = []
    self.scaler     = None
    self.time_ids   = []

    self.compute_representation = representation
    self.compute_semantic = semantic

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
      pairs.append([self.data[i], self.data[i+1]])
      tdelta = self.timestamps[i] - self.timestamps[i+1]
      gaps.append(float(abs(tdelta.total_seconds())))
      if self.labels[i] == self.labels[i+1]:
        tags.append(0)
      else:
        tags.append(1)

    self.data = self.compute_representation(pairs) 
    self.labels = tags
    self.data = np.asarray(self.data, dtype=np.float32)
    self.labels = np.asarray(self.labels, dtype=np.int32)

    gaps = np.asarray(gaps, dtype=np.float32)
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)
    gaps -= mean_gap
    gaps /= std_gap

    data_len = len(self.data[0, 0])
    temp = []
    for i in range(len(self.data)):
      time_gap = np.full((1, data_len), gaps[i])
      temp.append(np.append(self.data[i], time_gap, axis=0)) 
    self.data = np.asarray(temp, dtype=np.float32)

    print('Entry shape: ' + str(self.data[0].shape))
    print('Data shape:  ' + str(self.data.shape))

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
      gaps.append(float(abs(tdelta.total_seconds())))
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
