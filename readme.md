## Installation

We have to create a Python environment in order to install all the dependencies
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-tk
sudo pip3 install -U virtualenv
```

Create a virtual enviornment and activate it
```bash
virtualenv --system-site-packages -p python3 ~/tfcpu
source ~/tfcpu/bin/activate 
```

Now, let's install Python packages
```bash
pip3 install --upgrade pip
pip3 install --upgrade numpy scipy scikit-learn matplotlib gensim distance
```

### TensorFlow

[TensorFlow](https://www.tensorflow.org/install/pip) installation page has instructions for different setups. Follow steps on the page for installing a GPU version. Should you need a CPU based installation, run the following command

```bash
pip3 install --upgrade tensorflow
```

### FastText for Python
We need to [install](https://github.com/facebookresearch/fastText/#building-fasttext-for-python) fasttext so we can use the GenSim wrapper from Python
```bash
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip3 install .
```

### Running experiments
A Python script with experiments encodes the parameters for the experiments and allows to easily replicate results. Based on the content of this script, modifications can be performed for additional testing. For running them, activate the virtual environment and run the script

```
source ~/tfcpu/bin/activate 
python3 experiments.py
```

You might want to save the command line output to a log file

```
python3 experiments.py |& tee results.log
```