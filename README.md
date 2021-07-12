# CCNLab

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/nikhilxb/ccnlab/blob/classical/LICENSE)

**CCNLab** (short for Cognitive Computational Neuroscience Lab) is a benchmark for evaluating computational neuroscience models on empirical data. With classical conditioning as a case study, it includes a collection of seminal experiments in classical conditioning written in a domain-specific language, a common API for simulating different models on these experiments, and tools for visualizing and comparing the simulated data from the models with the empirical data.

CCNLab is designed to be: 

- **broad**, covering many different phenomena;
- **flexible**, allowing the straightforward addition of new experiments; and
- **easy to use**, so researchers can focus on developing better models.

We envision CCNLab as a testbed for unifying computational theories of learning in the brain. We also hope that it can broadly accelerate neuroscience research and facilitate interaction between the fields of neuroscience, psychology, and artificial intelligence.

Please refer to our paper for more background, technical details, and baseline results on a selection of existing models.

## Installation

1. Ensure you have Anaconda installed.
2. Clone this repository and set up the included Anaconda environment, which installs Python 3.7.10 and pip dependencies (including Jupyter Notebook).

```bash
conda env create -f environment.yml
conda activate ccnlab
```
3. To start Jupyter run:

```bash
jupyter notebook
```

## Usage

The provided Jupyter notebook `ClassicalConditioning.ipynb` provides working examples of how to simulate experiments, evaluate model results, and extend the benchmark with additional experiments. 
