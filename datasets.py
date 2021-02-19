"""
Load data from local folder and encode information according to the selected
representation
"""

import download_datasets

import io, csv, gc, os, sys, distance, json, time
import re, urllib.request, urllib.parse
from datetime import timezone, datetime

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np 
if tf.__version__.startswith('2.'):
  import scann
  import bert
else:
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

    self.max_seq_length = 32
    self.query_list_step = 5000

    self.cache_sep  = '|\t|\t|'
    self.muse_cache_file = download_datasets.DATA_DIR + '/muse/cache_muse.csv'
    self.muse_cache = {}
    self.labse_cache_file = download_datasets.DATA_DIR+'/labse/cache_labse.csv'
    self.labse_cache = {}

  def clear_cache(self):
    """
    Remove dictionaries for representation caches
    """
    self.muse_cache = {}
    self.labse_cache = {}

  def store_cache(self, cache_file, cache):
    """
    Store cache into local disk using a CSV file
    """
    dir_path = os.path.dirname(cache_file)
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

    with open(cache_file, mode='w', encoding='utf-8') as data_file:
      for query in cache:
        embedding = [str(number) for number in cache[query]]
        embedding = ','.join(embedding)
        data_file.write(self.cache_sep.join([query, embedding]) + '\n')
        
  def load_cache(self, cache_file, cache):
    """
    Load CSV file from local disk and store query embeddings into cache 
    local variable
    """
    if not os.path.exists(cache_file):
      return

    print('\tLoading cache ' + cache_file)
    with open(cache_file, mode='r') as data_file:
      for line in data_file:
        row = line.split(self.cache_sep)
        query = row[0]
        try :
          embedding = [float(number) for number in row[1].split(',')]
          cache[query] = embedding
        except Exception as e:
          print('\trow {} in cache {} does not have two fields, {}'.format(
            str(row), cache_file, str(e)))

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
    self.store_cache(self.muse_cache_file, self.muse_cache)
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
      self.load_cache(self.muse_cache_file, self.muse_cache)

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

  def tokenize_queries(self, queries):
    """
    User BERT tokenizer to compute tokens for queries

    [1] https://tfhub.dev/google/LaBSE/1
    """
    input_ids_all, input_mask_all, segment_ids_all = [], [], []
    for input_string in queries:
      input_tokens = \
        ["[CLS]"] + self.tokenizer.tokenize(input_string) + ["[SEP]"]
      input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
      sequence_length = min(len(input_ids), self.max_seq_length)

      # Padding or truncation.
      if len(input_ids) >= self.max_seq_length:
        input_ids = input_ids[:self.max_seq_length]
      else:
        input_ids = input_ids + [0] * (self.max_seq_length - len(input_ids))

      input_mask = \
        [1] * sequence_length + [0] * (self.max_seq_length - sequence_length)

      input_ids_all.append(input_ids)
      input_mask_all.append(input_mask)
      segment_ids_all.append([0] * self.max_seq_length)

    return np.array(
      input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)

  def load_labse_model(self):
    """
    Use tensorflow hub to load the pretrained LABSE model. After retrieving
    resolved objects for vocab and do_lower_case, load the tokenizer using 
    bert-for-tf2 bert module 

    [1] https://tfhub.dev/google/LaBSE/1
    """
    if self.vectors == None:
      labse_layer = hub.KerasLayer(
        download_datasets.LANGUAGE_AGNOSTIC_SENTENCE_E, trainable=False)
      input_word_ids = tf.keras.layers.Input(
        shape=(self.max_seq_length,), dtype=tf.int32, name="input_word_ids")
      input_mask = tf.keras.layers.Input(
        shape=(self.max_seq_length,), dtype=tf.int32, name="input_mask")
      segment_ids = tf.keras.layers.Input(
        shape=(self.max_seq_length,), dtype=tf.int32, name="segment_ids")
      pooled_output,  _ = labse_layer(
        [input_word_ids, input_mask, segment_ids])
      pooled_output = tf.keras.layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)
      self.vectors = tf.keras.Model(
        inputs=[input_word_ids, input_mask, segment_ids], outputs=pooled_output)

      vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()
      do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
      self.tokenizer = \
        bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

  def get_labse_vectors(self, queries):
    """
    Load query embeddings from the LABSE cache. If any query is not stored 
    there, load the LABSE model and update the cache
    """
    embeddings = []
    update_cache = False
     
    for query in queries:
      embedding = self.labse_cache.get(query, [])
      if len(embedding) == 0:
        update_cache = True
        break
      embeddings.append(embedding)

    if update_cache == False:
      return embeddings

    self.load_labse_model()
    input_ids, input_mask, segment_ids = self.tokenize_queries(queries)
    input_ids   = tf.cast(input_ids, dtype=tf.int32)
    input_mask  = tf.cast(input_mask, dtype=tf.int32)
    segment_ids = tf.cast(segment_ids, dtype=tf.int32)
    embeddings  = self.vectors([input_ids, input_mask, segment_ids])
    embeddings  = np.asarray(embeddings, dtype=np.float32)
    for i in range(len(queries)):
      self.labse_cache[queries[i]] = embeddings[i]
    self.store_cache(self.labse_cache_file, self.labse_cache)

    return embeddings

  def get_labse_vectors_nocache(self, queries):
    """
    Load the LABSE model and compute query vectors. Do not store information in
    the cache.
    """
    embeddings = []
    self.load_labse_model()

    for index in range(0, len(queries), self.query_list_step):
      input_ids, input_mask, segment_ids = self.tokenize_queries(
        queries[index : index + self.query_list_step])
      input_ids   = tf.cast(input_ids, dtype=tf.int32)
      input_mask  = tf.cast(input_mask, dtype=tf.int32)
      segment_ids = tf.cast(segment_ids, dtype=tf.int32)
      labse_embeddings = self.vectors([input_ids, input_mask, segment_ids])
      labse_embeddings = np.asarray(labse_embeddings, dtype=np.float32)
      embeddings.extend(labse_embeddings)
      
    return embeddings

  def labse(self, text_data):
    """
    Pretrained language agnostic BERT sentence embedding model

    [1] https://ai.googleblog.com/2020/08/language-agnostic-bert-sentence.html
    """
    embeddings = []

    if len(self.labse_cache) == 0:
      self.load_cache(self.labse_cache_file, self.labse_cache)

    if isinstance(text_data[0], list):
      for queries in text_data:
        labse_embeddings = self.get_labse_vectors(queries)
        embeddings.append(labse_embeddings)
    else:
      for index in range(0, len(text_data), self.query_list_step):
        labse_embeddings = self.get_labse_vectors(
          text_data[index : index + self.query_list_step])
        embeddings.extend(labse_embeddings)
        print('\tcomputing LABSE encodings, index {}'.format(index), flush=True)
    
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

class orcas(object):
  """
  Open Resource for Click Analysis in Search

  [1] https://microsoft.github.io/TREC-2020-Deep-Learning/ORCAS.html
  """
  def __init__(self, representation=representation().glove):
    self.CACHE_FILE = download_datasets.DATA_DIR + '/orcas/cache_ids.csv'
    self.CACHE_SEP  = '|\t|\t|'
    self.doc_ids = {}
    self.model = None
    self.top_k = 1000
    self.scann_leaves = 2000

    self.tsv_file = '/'.join(
      [download_datasets.DATA_DIR, 'orcas', 'orcas.tsv'])
    self.tsv_queries = None
    self.tsv_doc_ids = None
    self.representation=representation

  def load_cache(self):
    """
    Load TSV file from local disk and store query JSON results into cache 
    local variable
    """
    if not os.path.exists(self.CACHE_FILE):
      return

    print('\tLoading Orcas cache ' + self.CACHE_FILE)
    with open(self.CACHE_FILE, mode='r') as data_file:
      for line in data_file:
        row = line.split(self.CACHE_SEP)
        if len(row) == 2:
          self.doc_ids[row[0]] = row[1].split(',')
        else:
          print('\trow {} in cache {} does not have two fields'.format(
            str(row), self.CACHE_FILE))

  def save_cache(self):
    """
    Store cache into local disk using a CSV file
    """
    dir_path = os.path.dirname(self.CACHE_FILE)
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

    with open(self.CACHE_FILE, mode='w') as data_file:
      for query in self.doc_ids:
        ids = ','.join(self.doc_ids[query])
        data_file.write(self.CACHE_SEP.join([query, ids]) + '\n')

  def load_tsv(self):
    """
    Load queries and document IDs form the ORCAS tsv file, encoding the queries
    in the LABSE latent space with a local representation object
    """
    self.tsv_queries = []
    self.tsv_doc_ids = []
    with open(self.tsv_file, mode='r') as data_file:
      reader = csv.reader(data_file, delimiter='\t')
      for row in reader:
        self.tsv_queries.append(row[1])
        self.tsv_doc_ids.append(row[2])

    encoding = representation()
    encoding.query_list_step = 50000
    encoding.labse_cache_file = \
      download_datasets.DATA_DIR + '/labse/cache_orcas_labse.csv'
    self.tsv_queries = encoding.labse(self.tsv_queries)
    self.tsv_queries = np.asarray(self.tsv_queries, dtype=np.float32)

  def retrieve(self, query):
    """
    Build an ORCAS index using ScaNN. Retrieve the top k document IDs in
    the semantic space

    [1] https://github.com/google-research/google-research/tree/master/scann
    """
    if self.model == None:
      print('\tCreating Orcas ScaNN index ...')
      self.load_tsv()
      print('\n\tfinished query encoding', flush=True)
      self.model = scann.ScannBuilder(
        self.tsv_queries, self.top_k, 'dot_product').tree(
        num_leaves=4000, 
        num_leaves_to_search=self.scann_leaves, 
        training_sample_size=1000000).score_ah(
        dimensions_per_block=2, 
        anisotropic_quantization_threshold=0.2).reorder(
        reordering_num_neighbors=self.scann_leaves).create_pybind()
      print('\tfinished Orcas ScaNN index', flush=True)

    start = time.time()
    encoded_query = self.representation([query])[0]
    results, _ = self.model.search(
      encoded_query, final_num_neighbors=self.top_k)
    ids = []
    for index in results:
      ids.append(self.tsv_doc_ids[index])
    print('\tquery: {} time: {}'.format(query, time.time() - start))

    return ids

  def retrieve_document_ids_cache(self, query):
    """
    Return an array of document ids retrieved from the ORCAS index. If the query
    is not in the cache, load the whole index and update the cache
    """
    ids = self.doc_ids.get(query, [])
    if ids != []:
      return ids

    ids = self.retrieve(query)
    self.doc_ids[query] = ids
    self.save_cache()
    return ids

  def retrieve_document_ids(self, query):
    """
    Return an array of document ids retrieved from the ORCAS index
    """
    ids = self.doc_ids.get(query, [])
    if ids != []:
      return ids

    print('\tquery: {}, is not in the cache'.format(query))
    return ids

  def intent_similarity_ids(self, q0, q1):
    """
    Use the IDs from the top documents retrieved from ORCAS index to 
    calculate the intent similarity

    [1] AOLTaskExtraction/src/LucheseImplementation/SimScoreCalculation.java
    """
    if q0 == q1:
      return 1.0
    
    if len(self.doc_ids) == 0:
      self.load_cache()

    q0_ids = self.retrieve_document_ids(q0)
    q1_ids = self.retrieve_document_ids(q1)

    initial_size = len(q0_ids)
    for id in q1_ids:
      if id in q0_ids:
        q0_ids.remove(id)

    intersec_size = initial_size - len(q0_ids)
    if len(q0_ids) != 0 and len(q1_ids) != 0:
      return float(intersec_size) / (
        float(initial_size + len(q1_ids) - intersec_size))

    return 0.0

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
    self.file_encoding = 'utf8'
    self.augmented_delimiter  = '|\t|\t|'
    self.augmented_file = '/'.join(
      [download_datasets.DATA_DIR, 'offline_back_translate.csv'])
    self.aumgented_data = []

    self.compute_representation = representation

  def load(
    self, textdata=False, prefix_title=None, data_idx=0, label_idx=1, 
    filter_idx=2, filter_set=None):
    """
    Load data into object variables according to the specified representation.
    If textdata is True, do not compute any representation for the queries. 
    Use filter_idx field to ignore entries in the CSV with values in the 
    filter_set
    """
    label_dict = {}

    with open(self.file, mode='r', encoding=self.file_encoding) as data_file:
      reader = csv.reader(data_file, delimiter=self.delimiter)
      try:
        for row in reader:
          if prefix_title != None and row[0].startswith(prefix_title):
            continue
          if filter_set != None and row[filter_idx] in filter_set:
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
        print('Exception while loading dataset ' + self.file)
        print('Loaded %d queries and %d labels \n%s'%(
          len(self.data), len(self.labels), e))
        exit()

    self.labels = np.asarray(self.labels, dtype=np.int32)
    if textdata == False:
      self.data = self.compute_representation(self.data)
      self.data = np.asarray(self.data, dtype=np.float32)
      print("\tData size: " + str(self.data.shape))

  def load_augmented(self, textdata=False):
    """
    Use back tranlation offline dictionary to load augmented queries for queries
    in self.data
    """
    augmentation = dict()
    with open(
      self.augmented_file, mode='r', encoding=self.file_encoding) as data_file:
      for line in data_file:
        row = line.strip().split(self.augmented_delimiter)
        augmentation[row[0]] = row[1]

    for query in self.data:
      self.aumgented_data.append(augmentation.get(query, ''))

    if textdata == False:
      self.data = self.compute_representation(self.data)
      self.data = np.asarray(self.data, dtype=np.float32)
      self.aumgented_data = self.compute_representation(self.aumgented_data)
      self.aumgented_data = np.asarray(self.aumgented_data, dtype=np.float32)
      print("\tData size: " + str(self.data.shape))
      print("\tData size augmented: " + str(self.aumgented_data.shape))

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

  def load_filter_user(self, textdata=False):
    """
    Consider only users' queries, ignoring data augmentation entries from 
    Google and Bing
    """
    super().load(
      textdata=textdata, prefix_title='Query', data_idx=0, label_idx=2,
      filter_idx=1, filter_set=set(['google', 'bing'])) 

  def load_augmented_filter_user(self, textdata=False):
    """
    Consider only users' queries, ignoring data augmentation entries from 
    Google and Bing
    """
    self.load_filter_user(textdata=True)
    super().load_augmented(textdata=textdata)

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

  def load_filter_user(self, textdata=False):
    """
    Consider only users' queries, ignoring data augmentation entries from 
    Google and Bing
    """
    super().load(
      textdata=textdata, prefix_title='Query', data_idx=0, label_idx=2,
      filter_idx=1, filter_set=set(['google', 'bing'])) 

  def load_augmented_filter_user(self, textdata=False):
    """
    Consider only users' queries, ignoring data augmentation entries from 
    Google and Bing
    """
    self.load_filter_user(textdata=True)
    super().load_augmented(textdata=textdata)

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

  def load_filter_user(self, textdata=False):
    """
    Consider only users' queries, ignoring data augmentation entries from 
    Google and Bing
    """
    super().load(
      textdata=textdata, prefix_title='Query', data_idx=0, label_idx=2,
      filter_idx=1, filter_set=set(['google', 'bing'])) 

  def load_augmented_filter_user(self, textdata=False):
    """
    Consider only users' queries, ignoring data augmentation entries from 
    Google and Bing
    """
    self.load_filter_user(textdata=True)
    super().load_augmented(textdata=textdata)

class wp4_task(csv_base_dataset):
  """
  Search task dataset from Dosso et al., 2020
  """
  def __init__(self, representation=representation().glove):
    super().__init__(representation)
    self.delimiter = '\t'
    self.file = '/'.join(
      [download_datasets.DATA_DIR, 'wp4', 'wp4_tasks.csv'])

  def load(self, textdata=False):
    super().load(
      textdata=textdata, prefix_title=None, data_idx=0, label_idx=1)

  def load_augmented(self, textdata=False):
    self.load(textdata=True)
    super().load_augmented(textdata=textdata)
    
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

  def load_augmented(self, textdata=False):
    self.load(textdata=True)
    super().load_augmented(textdata=textdata)

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

  def load_sequential_pair_dual(self):
    """
    Load data in sequential pairs of queries, tagging with a one
    when there is a task change, zero otherwise. Instead of using one vector 
    per query, use multiple embeddings, loading the queries separetely into 
    left and right arrays
    """

    self.read_csv()

    left_queries, right_queries, tags, gaps = [], [], [], []
    for i in range(len(self.data) - 1):
      left_queries.append(self.data[i])
      right_queries.append(self.data[i+1])
      tdelta = self.timestamps[i] - self.timestamps[i+1]
      gaps.append(float(abs(tdelta.total_seconds())))
      if self.labels[i] == self.labels[i+1]:
        tags.append(0)
      else:
        tags.append(1)

    self.data_left = self.compute_representation(left_queries)
    self.data_right = self.compute_representation(right_queries)
    self.labels = tags
    self.data_left = np.asarray(self.data_left, dtype=np.float32)
    self.data_right = np.asarray(self.data_left, dtype=np.float32)
    self.labels = np.asarray(self.labels, dtype=np.int32)

    print('Entry shape: ' + str(self.data_left[0].shape) \
      + ' ' + str(self.data_right[0].shape))
    print('Data shape:  ' + str(self.data_left.shape) \
      + ' ' + str(self.data_right.shape))

  def load_random_pair_dual(self):
    """
    Load data in pairs of queries, tagging with a one when they are sequential, 
    zero otherwise. Instead of using one vector per query, use multiple 
    embeddings, loading the queries separately into left and right arrays
    """

    self.read_csv()

    left_queries, right_queries, tags, gaps = [], [], [], []
    for i in range(len(self.data) - 1):
      left_queries.append(self.data[i])
      right_queries.append(self.data[i+1])
      tags.append(1)

    for i in range(len(self.data) - 1):
      idxs =  np.random.randint(len(self.data), size=2)
      while np.abs(idxs[0] - idxs[1]) == 1:
        idxs =  np.random.randint(len(self.data), size=2)
      left_queries.append(self.data[idxs[0]])
      right_queries.append(self.data[idxs[1]])
      tags.append(0)

    self.data_left = self.compute_representation(left_queries)
    self.data_right = self.compute_representation(right_queries)
    self.labels = tags
    self.data_left = np.asarray(self.data_left, dtype=np.float32)
    self.data_right = np.asarray(self.data_left, dtype=np.float32)
    self.labels = np.asarray(self.labels, dtype=np.int32)

    print('Entry shape: ' + str(self.data_left[0].shape) \
      + ' ' + str(self.data_right[0].shape))
    print('Data shape:  ' + str(self.data_left.shape) \
      + ' ' + str(self.data_right.shape))

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
