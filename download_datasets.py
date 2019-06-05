"""
Download datasets, uncompress, and store them under the datasets folder
"""

import os, re, urllib, urllib.request, gzip, tarfile, zipfile, shutil

# Constants
DATA_DIR = "datasets"

AOL_LUCCHESE = "http://miles.isti.cnr.it/~tolomei/?download=aol-task-ground-truth.tar.gz"
AOL_PROCHETA = "https://raw.githubusercontent.com/procheta/AOLTaskExtraction/master/Task.csv"
AOL_GAYO_AVELLO = "http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-smc-12/corpus-webis-smc-12.zip"
SOGOUQ = "https://www.sogou.com/labs/resource/ftp.php?dir=/Data/SogouQ/SogouQ.tar.gz"

REUTERS10K = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/"
REUTERS10K_FILES = [
  REUTERS10K + "a12-token-files/lyrl2004_tokens_test_pt0.dat.gz", 
  REUTERS10K + "a12-token-files/lyrl2004_tokens_test_pt1.dat.gz",
  REUTERS10K + "a12-token-files/lyrl2004_tokens_test_pt2.dat.gz", 
  REUTERS10K + "a12-token-files/lyrl2004_tokens_test_pt3.dat.gz",
  REUTERS10K + "a12-token-files/lyrl2004_tokens_train.dat.gz",
  REUTERS10K + "a08-topic-qrels/rcv1-v2.topics.qrels.gz"
]

GLOVE = "http://nlp.stanford.edu/data/glove.42B.300d.zip"
FASTTEXT_EN = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
# Manual download from Google Drive
WORD2VEC_EN = "https://drive.google.com/uc?export=download&confirm=YjKc&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"

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

def download_files():
  """
  Download text datasets 
  """

  download_file(
    '/'.join([DATA_DIR, "aol"]), "aol-task-ground-truth.tar.gz", AOL_LUCCHESE)
  download_file(
    '/'.join([DATA_DIR, "aol"]), "procheta_task.csv", AOL_PROCHETA)
  download_file(
    '/'.join([DATA_DIR, "aol"]), "corpus-webis-smc-12.zip", AOL_GAYO_AVELLO)
  download_file(
    '/'.join([DATA_DIR, "fasttext"]), "cc.en.300.bin.gz", FASTTEXT_EN)
  download_file(
    '/'.join([DATA_DIR, "word2vec"]), "GoogleNews-vectors-negative300.bin.gz", 
    WORD2VEC_EN)
  download_file(
    '/'.join([DATA_DIR, "glove"]), "glove.42B.300d.zip", GLOVE)
  
  for url in REUTERS10K_FILES:
    filename = os.path.basename(url)
    download_file('/'.join([DATA_DIR, "reuters"]), filename, url)

if __name__ == "__main__":
  download_files()