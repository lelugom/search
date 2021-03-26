## Modelling patterns of search behaviours from user interactions

## Installation

We have to create a Python environment in order to install all the dependencies
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-tk
sudo pip3 install -U virtualenv
```

Create a virtual environment and activate it
```bash
virtualenv --system-site-packages -p python3 ~/tfcpu
source ~/tfcpu/bin/activate 
```

Now, let's install Python packages
```bash
pip3 install --upgrade pip
pip3 install --upgrade numpy scipy scikit-learn keras 
pip3 install --upgrade matplotlib distance absl-py pyemd 
pip3 install --upgrade pygtrie==2.3.2 networkx==2.3 ngt==1.11.6 gensim==3.8.1
```

Additional installation packages will depend on the experiments to replicate. Every section has installation instructions. 

#### Containers

To facilitate code execution, two [Docker](https://www.docker.com/) containers are available with all dependencies installed, select one of them depending on the experiment to run

- [tf1_12_cpu](https://hub.docker.com/repository/docker/lelugom/tf1_13_cpu) for experiments requiring TensorFlow 1.13.1. This version is for CPU only. The container is based on the NVidia TensorFlow [container](nvcr.io/nvidia/tensorflow:20.03-tf1-py3)
- [tf2_gcc9](https://hub.docker.com/repository/docker/lelugom/tf2_gcc9) for experiments requiring TensorFlow 2.1.0 and Scalable Nearest Neighbors. The container is based on the Docker TensorFlow [container](https://hub.docker.com/r/tensorflow/tensorflow/)

These containers can easily run on platforms like [Singularity](https://sylabs.io/docs/). For example, running a modeling experiment with the [tf2_gcc9](https://hub.docker.com/repository/docker/lelugom/tf2_gcc9) container is quite simple 
```bash
singularity exec docker://lelugom/tf2_gcc9 python3 experiments.py -t modeling
```

## Datasets
Multiple datasets are supported for the methods in this repository. The download_datasets.py Python module has both the URLs and the code to download and decompress the datasets. Run the module to download and decompress the datasets

```bash
python3 download_datasets.py
```

#### Complex User Search Task Analysis (CUSTA) dataset

In the _datasets/wp4 _folder you can find the _wp4_tasks.csv_ dataset, a collection of queries with ground-truth labels for search tasks. Each entry contains two columns:

- Query: effet du bruit et du sommeil sur la concentration
- Label: 3

Reference
```
@inproceedings{dosso2020how,
  title={How to support search activity of users without prior domain knowledge when they are solving learning tasks?},
  author={Dosso, Cheyenne and Chevalier, Aline and Tamine, Lynda},
  booktitle={Proceedings of the 29th ACM International Conference on Information and Knowledge Management},
  series={1st International Workshop on Investigating Learning During Web Search},
  year={2020},
  organization={ACM}
}
```

## Running experiments
A Python script encodes the parameters for the experiments and allows the easy replication of results. Based on the content of this script, modifications can be performed for additional testing. Bear in mind that experiments have multiple repetitions for cross-validation and statistical significance tests; thus, execution times can be long depending on your machine. 

### Segmenting Search Query Logs

Install [TensorFlow](https://www.tensorflow.org/install/pip), [TensorFlow Hub](https://medium.com/tensorflow/introducing-tensorflow-hub-a-library-for-reusable-machine-learning-modules-in-tensorflow-cdee41fa18f9), a library for easy sharing of trained machine learning models, and [SentencePiece](https://github.com/google/sentencepiece) for calculating sentence encodings 

```bash
pip3 install tensorflow==1.13.1 tensorflow-hub==0.5.0 
pip3 install sentencepiece==0.1.83 tf-sentencepiece==0.1.83
```

For replicating session segmentation results, run the script

```bash
python3 experiments.py -t segmentation
```
Reference
```
@inproceedings{lugo2020segmenting,
  title={Segmenting Search Query Logs by Learning to Detect Search Task Boundaries},
  author={Lugo, Luis and Moreno, Jose G. and Hubert, Gilles},
  booktitle={The 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020}
}
```

### Multilingual search task identification

Install [TensorFlow](https://www.tensorflow.org/install/pip), [TensorFlow Hub](https://medium.com/tensorflow/introducing-tensorflow-hub-a-library-for-reusable-machine-learning-modules-in-tensorflow-cdee41fa18f9), and [SentencePiece](https://github.com/google/sentencepiece) 

```bash
pip3 install tensorflow==1.13.1 tensorflow-hub==0.5.0 
pip3 install sentencepiece==0.1.83 tf-sentencepiece==0.1.83
```

For replicating multilingual task identification results, run the script

```bash
python3 experiments.py -t identification
```

Use the automatic translation tool of your choice - Google Cloud has a translation [API](https://cloud.google.com/translate) - to translate the queries in the file _datasets/aol/procheta_task.csv_ and rerun the script to compute results in other languages. Results using an external index for query similarity calculations require the [ClueWeb12](https://lemurproject.org/clueweb12/) dataset. Follow instructions on the webpage to download the dataset. Serve the top one thousand entries using  JSON and UTF-8 at an endpoint supporting HTTP GET. The GET endpoint only requires one name/value pair (query=query_text). Click [here](http://clueweb.adaptcentre.ie/WebSearcher/search?query=tree) to see an example of results for the query "tree". Once you have the URL for the endpoint, run the script

```bash
python3 experiments.py -t identification -u clueweb_url
```
Reference
```
@inproceedings{lugo2020multilingual,
  title={A Multilingual Approach for Unsupervised Search Task Identification},
  author={Lugo, Luis and Moreno, Jose G. and Hubert, Gilles},
  booktitle={The 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020}
}
```

### Recurrent deep clustering for search task extraction

Install [TensorFlow](https://www.tensorflow.org/install/pip), [TensorFlow Hub](https://medium.com/tensorflow/introducing-tensorflow-hub-a-library-for-reusable-machine-learning-modules-in-tensorflow-cdee41fa18f9), and [SentencePiece](https://github.com/google/sentencepiece)  

```bash
pip3 install tensorflow==1.13.1 tensorflow-hub==0.5.0 
pip3 install sentencepiece==0.1.83 tf-sentencepiece==0.1.83
```

For replicating recurrent deep clustering results, run the script

```bash
python3 experiments.py -t extracting
```
Reference
```
@inproceedings{lugo2021extracting,
  title={Extracting Search Tasks from Query Logs Using a Recurrent Deep Clustering Architecture},
  author={Luis Lugo and Jose G. Moreno and Gilles Hubert},
  booktitle={Proceedings of the 43rd European Conference on Information Retrieval},
  year={2021},
  organization={Springer}
}
```

### Language-agnostic modeling of user search tasks

These modeling experiments requires [Scalable Nearest Neighbors](https://github.com/google-research/google-research/tree/master/scann) (ScaNN) for user search intent and  query task mapping. First, install GCC 9

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-9 g++-9 wget
```

Install [TensorFlow](https://www.tensorflow.org/install/pip), [TensorFlow Hub](https://medium.com/tensorflow/introducing-tensorflow-hub-a-library-for-reusable-machine-learning-modules-in-tensorflow-cdee41fa18f9), [TensorFlow Text](https://github.com/tensorflow/text), and [BERT for TF2](https://pypi.org/project/bert-for-tf2/) for the tokenizer 

```bash
pip3 install bert-for-tf2==0.14.6 
pip3 install tensorflow==2.1.1 tensorflow-text==2.1.1 tensorflow-hub==0.9.0
```

Finally, install ScaNN
```bash
wget https://storage.googleapis.com/scann/releases/1.0.0/scann-1.0.0-cp36-cp36m-linux_x86_64.whl
pip3 install scann-1.0.0-cp36-cp36m-linux_x86_64.whl
```

For replicating language-agnostic modeling results, run the script
```bash
python3 experiments.py -t modeling
```
As the ORCAS dataset has 18 million queries, a cache of document IDs is stored in the file _datasets/orcas/cache_ids.csv_. The default configuration loads this cache to avoid building the ORCAS index in the RAM while modeling the search tasks. However, if you want to build the ORCAS index - bear in mind that it consumes a lot of RAM - run the script

```bash
python experiments.py -t modeling -u orcas_cache
```

Reference
```
@inproceedings{lugo2021modeling,
  title={Modeling User Search Tasks with a Language-agnostic Unsupervised Approach},
  author={Luis Lugo and Jose G. Moreno and Gilles Hubert},
  booktitle={Proceedings of the 43rd European Conference on Information Retrieval},
  year={2021},
  organization={Springer}
}
```

### License
This project is licensed under the terms of the [MIT](https://opensource.org/licenses/MIT) license.
