"""
Download datasets, uncompress, and store them under the datasets folder
"""

import os, re, urllib, urllib.request, gzip, glob, tarfile, zipfile, shutil
import tensorflow as tf

# Constants
DATA_DIR = "datasets"

AOL_LEAK = 'https://archive.org/download/AOL_search_data_leak_2006/AOL_search_data_leak_2006.zip'
AOL_LUCCHESE = "http://pomino.isti.cnr.it/~tolomei/downloads/aol-task-ground-truth.tar.gz"
AOL_SEN = "https://raw.githubusercontent.com/procheta/AOLTaskExtraction/master/Task.csv"
AOL_HAGEN = "https://zenodo.org/record/3265962/files/corpus-webis-smc-12.zip?download=1"
VOLSKE = "https://zenodo.org/record/3257431/files/webis-qtm-19.zip?download=1"
ORCAS  = 'https://msmarco.blob.core.windows.net/msmarcoranking/orcas.tsv.gz'

GLOVE = "http://nlp.stanford.edu/data/glove.42B.300d.zip"
if tf.__version__.startswith('2.'):
  M_UNIVERSAL_SENTENCE_ENCODER = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
else:
  M_UNIVERSAL_SENTENCE_ENCODER = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1'
LANGUAGE_AGNOSTIC_SENTENCE_E = 'https://tfhub.dev/google/LaBSE/1'
MUSE_LANGS = ['ar', 'zh', 'zh-tw', 'nl', 'en', 'de', 'fr', 'it', 'pt', 'es', 
  'ja', 'ko', 'ru', 'pl', 'th', 'tr']

def uncompress(folder, file):
  """
  Uncompress tar.gz, zip, and gz files
  """

  if file.endswith('.tar.gz'):
    untar = tarfile.open(file, 'r:gz')
    untar.extractall(path=folder)
    untar.close()
  elif file.endswith('.zip'):
    unzip = zipfile.ZipFile(file, 'r')
    unzip.extractall(path=folder)
    unzip.close()
  elif file.endswith('.gz'):
    with gzip.open(file, 'rb') as file_in:
      with open(re.compile(r'\.gz$').sub('', file), 'wb') as file_out:
          shutil.copyfileobj(file_in, file_out)

def download_file(folder, filename, url):
  """
  Download file from url and store under folder/filename. Uncompress if needed
  """

  file = '/'.join([folder, filename])

  if not os.path.exists(folder):
    print('Could not find %s, creating now...' % folder)
    os.makedirs(folder)

  if not os.path.exists(file):
    print('Retrieving: %s' % filename)
    urllib.request.urlretrieve(url, file)

  uncompress(folder, file)

def consolidate_aol_leak():
  """
  After downloading and decompressing dataset file, clean aol directory,
  decompress collection zip files and consolidate all the txts into a single
  file

  [1] https://stackoverflow.com/questions/17749484/python-script-to-concatenate-all-the-files-in-the-directory-into-one-file
  """
  folder = '/'.join([DATA_DIR, 'aol', 'AOL-user-ct-collection'])
  if os.path.isdir('/'.join([DATA_DIR, 'aol', '__MACOSX'])):
    shutil.rmtree('/'.join([DATA_DIR, 'aol', '__MACOSX']))
  
  outfilename = folder + '/user-ct-test-collection.txt'
  with open(outfilename, 'wb') as outfile:
    for filename in sorted(glob.glob(folder + '/*.gz')):
      uncompress('', filename)
      with open(filename.replace('.txt.gz', '.txt'), 'rb') as readfile:
        shutil.copyfileobj(readfile, outfile)
        
def download_files():
  """
  Download text datasets 
  """
  download_file(
    '/'.join([DATA_DIR, "aol"]), "AOL_search_data_leak_2006.zip", AOL_LEAK)
  download_file(
    '/'.join([DATA_DIR, "aol"]), "aol-task-ground-truth.tar.gz", AOL_LUCCHESE)
  download_file(
    '/'.join([DATA_DIR, "aol"]), "procheta_task.csv", AOL_SEN)
  download_file(
    '/'.join([DATA_DIR, "aol"]), "corpus-webis-smc-12.zip", AOL_HAGEN)
  download_file(
    '/'.join([DATA_DIR, "volske"]), "webis-qtm-19.zip", VOLSKE)
  download_file(
    '/'.join([DATA_DIR, "glove"]), "glove.42B.300d.zip", GLOVE)
  download_file(
    '/'.join([DATA_DIR, "orcas"]), "orcas.tsv.gz", ORCAS)

  consolidate_aol_leak()

if __name__ == "__main__":
  download_files()