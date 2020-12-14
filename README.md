# Taxonomic Classification of NGS Reads using Language Models
This repository provides a script for the frame and taxnomic classification of NGS reads. It contains all resources needed to reproduce the results presentet in the work [Taxonomic Classification of NGS Reads using Language Models](https://github.com/CBMI-HTW/TaxonomicClassification-NGS-NN).

![Classification Results](https://redmine.f4.htw-berlin.de/owncloud/index.php/s/2WTqst8zckQdEQg/preview)


With the script `classification.py` you can classify your NGS-reads into the taxonomic domains viruses, bacteria, and mammals. The classification is done by concatenating multiple data processing and sub-classification steps. At first, the frame of a read within its coding sequence must be recognized to translate the DNA sequence fragments into amino acid sequences correctly. This is done by a transformer neural network. Using this information, the read sequences can be translated into amino acid sequences. In a final step, the amino acid sequences are classified into taxonomic domains by another transformer.


## Requirements

In order to run the scripts, you need to install the appropriate dependencies. We provide an `environment.yml` for conda and `environment.txt` for pip to clone the virtual environment used. 

```
# Create venv using conda
conda env create -f environment.yml

# Create venv using pip
pip install -r environment.txt
```

Note, since the environment contains a [PyTorch](https://pytorch.org/) version with gpu support (Cuda 11.0), these environments may not work for you depending on your hardware configuration.


## Usage
To classify your NGS reads, you can use the `classification.py` script. The script will take a file in [FASTA](https://en.wikipedia.org/wiki/FASTA_format) format as input and will output that file with an enriched header line (`>`) containing the results, i.e., ` source header line | STOP codon found? | frame classification | taxonomic classification`. At the moment the script expects the file to be clear of comments (`;`).

```
# Example of two lines from an input file
>gi|5524211|gb|AAD44166.1| cytochrome b [Elephas maximus maximus]
LCLYTHIGRNIYYGSYLYSETWNTGIMLLLITMATAFMGYVLPWGQMSFWGATVITNLFSAIPYIGTNLVEWIWGGFSVDKATLNRFFAFHFILPFTMVA

# Expected output after classification 
>gi|5524211|gb|AAD44166.1| cytochrome b [Elephas maximus maximus] | False | 2 | 0
LCLYTHIGRNIYYGSYLYSETWNTGIMLLLITMATAFMGYVLPWGQMSFWGATVITNLFSAIPYIGTNLVEWIWGGFSVDKATLNRFFAFHFILPFTMVA
```

Possible result values for the frame classification are 0-5 (0: on-frame, 1: offset by one base, 2: offset by two bases, 3: reverse-complementary, 4: reverse-complementary and offset by one base, 5: reverse complementary and offset by two bases). Possible classes for the taxonomic classification are 0: viral, 1: bacterial and 2: mammels.

The only argument required by the script is `--input` the path to the FASTA file you like to classify. All optional arguments can be viewed using the `--help` flag. First execution of the script will download the pre-trained models used in the process automatically. A few example calls are:

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
Final datasets and models (PyTorch version) used for the published experiments are public avaiable under Creative [Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) License. Executing the `classification.py` or `reproduce.py` script will trigger automatic download of the necessary models/datasets.

*Datasets*
- [Training: Frame Classification](https://zenodo.org/record/4306248)
- [Training: Taxonomic Classification](https://zenodo.org/record/4306240)
- [Testing: Taxonomic Classification](https://zenodo.org/record/4307779)

*PyTorch Models*
- [Frame Classification StateDict](https://zenodo.org/record/4306420)
- [Taxonomic Classification StateDict](https://zenodo.org/record/4306499)

## License
Distributed under the LGPL-2.1 License. See [LICENSE](LICENSE.md) for more information.
