## Modelling patterns of search behaviours from user interactions

### Installation

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
pip3 install --upgrade matplotlib gensim distance absl-py ngt pyemd networkx
```

### TensorFlow

[TensorFlow](https://www.tensorflow.org/install/pip) installation page has instructions for different setups. Follow steps on the page for installing a GPU version. For a CPU based installation, run the following command

```bash
pip3 install --upgrade tensorflow==1.13.1
```

After installing TensorFlow, we need to install [TensorFlow Hub](https://medium.com/tensorflow/introducing-tensorflow-hub-a-library-for-reusable-machine-learning-modules-in-tensorflow-cdee41fa18f9), a library for easy sharing of trained machine learning models

```bash
pip3 install --upgrade tensorflow-hub==0.5.0
```

### Multilingual USE

We also need [SentencePiece](https://github.com/google/sentencepiece) for calculating sentence encodings using the [multilingual](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1) universal sentence encoder

```bash
pip3 install sentencepiece==0.1.83 tf-sentencepiece==0.1.83
```

### Datasets
Multiple datasets are supported for the methods in this repository. The download_datasets.py Python module has both the URLs and the code to download and decompress the datasets. Run the module to download and decompress the datasets

```bash
python3 download_datasets.py
```

### Running experiments
A Python script encodes the parameters for the experiments and allows the easy replication of results. Based on the content of this script, modifications can be performed for additional testing. Bear in mind that experiments have multiple repetitions for cross-validation and statistical significance tests; thus, execution times can be long depending on your machine. First, activate the virtual environment 

```bash
source ~/tfcpu/bin/activate 
```

For replicating session segmentation results, run the script

```bash
python3 experiments.py -t segmentation
```

For replicating multilingual task identification results, run the script

```bash
python3 experiments.py -t identification
```

Use the automatic translation tool of your choice - Google Cloud has a translation [API](https://cloud.google.com/translate) - to translate the queries in the file _datasets/aol/procheta_task.csv_ and rerun the script to compute results in other languages. Results using an external index for query similarity calculations require the [ClueWeb12](https://lemurproject.org/clueweb12/) dataset. Follow instructions on the webpage to download the dataset. Serve the top one thousand entries using  JSON and UTF-8 at an endpoint supporting HTTP GET. The GET endpoint only requires one name/value pair (query=query_text). Click [here](http://clueweb.adaptcentre.ie/WebSearcher/search?query=tree) to see an example of results for the query "tree". Once you have the URL for the endpoint, run the script

```bash
python3 experiments.py -t identification -u clueweb_url
```

### Citation

```
@inproceedings{sesseg,
  title={Segmenting Search Query Logs by Learning to Detect Search Task Boundaries},
  author={Lugo, Luis and Moreno, Jose G. and Hubert, Gilles},
  booktitle={The 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020}
}
```
```
@inproceedings{taskide,
  title={A Multilingual Approach for Unsupervised Search Task Identification},
  author={Lugo, Luis and Moreno, Jose G. and Hubert, Gilles},
  booktitle={The 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020}
}
```

### License
This project is licensed under the terms of the [MIT](https://opensource.org/licenses/MIT) license.
