# Taxonomic Classification of NGS Reads using Language Models
This repository provides a script for the frame and taxnomic classification of NGS reads. It contains all resources needed to reproduce the results presentet in the work [Taxonomic Classification of NGS Reads using Language Models](https://github.com/CBMI-HTW/TaxonomicClassification-NGS-NN).

![Classification Results](https://redmine.f4.htw-berlin.de/owncloud/index.php/s/2WTqst8zckQdEQg/preview)


With the script `classification.py` you can classify your NGS-reads into the taxonomic domains viruses, bacteria, and mammals. The classification is done by concatenating multiple data processing and sub-classification steps. At first, the frame of a read within its coding sequence must be recognized to translate the DNA sequence fragments into amino acid sequences correctly. This is done by a transformer neural network. Using this information, the read sequences can be translated into amino acid sequences. In a final step, the amino acid sequences are classified into taxonomic domains by another transformer.


## Requirements
#### Using Poetry
In order to run the scripts, you need to install the appropriate dependencies. We use [poetry](https://python-poetry.org/) for denpedency management. We refer to peotry documentation for installtion instructions. Once poetry is installed you can install the dependecies with:

```bash
# Install dependencies
poetry install
```

Note, since the environment contains a [PyTorch](https://pytorch.org/) this setup is maybe not optimal for you. Since the standard pytorch package is not supporting _GPUs_, we strongly recommand replacing the [PyTorch](https://pytorch.org/) version with one that is suited for your enviroment and hardware configuration.
```
# You can replace the PyTorch version with one that suits your hardware configuration 
# by replacing the version in the poetry venv, e.g. GPU/CUDA11 support:
poetry shell
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Without Poetry
If you do not get the poetry version running, install following dependencies manually in a python virtual enviroment.

```bash
pip install pandas seaborn scikit-learn, bio

# Install a suitable PyTorch version, e.g. with CUDA 10.2 support. Use https://pytorch.org/ to find your suited version.
pip install torch torchvision torchaudio

# And final install packeges relying on PyTorch
pip install pytorch-nlp, transformers

```


## Usage
To classify your NGS reads, you can use the `classification.py` script. The script will take a file in [FASTA](https://en.wikipedia.org/wiki/FASTA_format) format as input and will output that file with an enriched header line (`>`) containing the results, i.e., ` source header line | STOP codon found? | frame classification | taxonomic classification`. At the moment, the script expects the file to be clear of comments (`;`).

```
# Example of two lines from an input file
>gi|5524211|gb|AAD44166.1| cytochrome b [Elephas maximus maximus]
LCLYTHIGRNIYYGSYLYSETWNTGIMLLLITMATAFMGYVLPWGQMSFWGATVITNLFSAIPYIGTNLVEWIWGGFSVDKATLNRFFAFHFILPFTMVA

# Expected output after classification 
>gi|5524211|gb|AAD44166.1| cytochrome b [Elephas maximus maximus] | False | 2 | 0
LCLYTHIGRNIYYGSYLYSETWNTGIMLLLITMATAFMGYVLPWGQMSFWGATVITNLFSAIPYIGTNLVEWIWGGFSVDKATLNRFFAFHFILPFTMVA
```

Possible result values for the frame classification are 0-5 (0: on-frame, 1: offset by one base, 2: offset by two bases, 3: reverse-complementary, 4: reverse-complementary and offset by one base, 5: reverse complementary and offset by two bases). Possible classes for the taxonomic classification are 0: viral, 1: bacterial and 2: mammals.

The only argument required by the script is `--input` the path to the FASTA file you like to classify. All optional arguments can be viewed using the `--help` flag. The First execution of the script will download the pre-trained models used in the process automatically. A few example calls are 

Note:**If you using poetry put a `poetry run` infront of the python call or activate the poetry venv `poetry shell`**

```
# Standard call
python classification.py -i /path/to/my/file.fasta

# In addtion to the FASTA file output create numpy files containing 
# the classification logits and a pandas dataframe
python classification.py -i /path/to/my/file.fasta -v

# Changing the the default output directory './results'
python classification.py -i /path/to/my/file.fasta -o /save/results/here
```


## Pre-Trained Models and Dataset
Final datasets and models (PyTorch version) used for the published experiments are publicly available under Creative [Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) License. Executing the provided `classification.py` or `reproduce_results.py` script files will automatically download the necessary models/datasets.

*Datasets*
- [Training: Frame Classification](https://zenodo.org/record/4306248)
- [Testing: Frame Classification](https://zenodo.org/record/4306248)
- [Training: Taxonomic Classification](https://zenodo.org/record/4306240)
- [Testing: Taxonomic Classification](https://zenodo.org/record/4307779)

*PyTorch Models*
- [Frame Classification StateDict](https://zenodo.org/record/4306420)
- [Taxonomic Classification StateDict](https://zenodo.org/record/4306499)


## Reproduce Paper Results
To reproduce the figures and accuracy values reported in the paper, you can execute the `reproduce_results.py`. The script will download all necessary resources (final models and test datasets) and calculate the used metrics. By default, all results will be saved in a folder called `paper-results`. Due to the size of the test datasets, the calculation time is relatively high. We recommend the usage of a GPU if you want to reproduce the results. The script's execution time using an NVIDIA RTX 3090 with a batch size of 1024 is about 5 hours.

```
# Standard call; use --help flag to see all arguments using poetry.
python reproduce_results.py

# Without poetry
python reproduce_results.py

```


## License
Distributed under the LGPL-2.1 License. See [LICENSE](LICENSE.md) for more information.
