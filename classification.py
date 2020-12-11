
import os
import sys
import argparse
import time
import json
from urllib.request import urlretrieve

import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.Seq import Seq

import torch
from torch.utils.data import DataLoader
from torchnlp.utils import collate_tensors
from transformers import BertTokenizer

from training.prot_bert.model import ProtTransClassification
from training.prot_bert.utils import accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Taxonomic Classification of NGS Reads using Language Models"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to Fasta file that should be classified",
        required=True
    )
    # General
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path where the evaluation results are stored",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="In addition to a output FASTA file numpy files with logits of the classifiers and a pandas with result informations are written to the output directory."
    )
    # Training
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Script will check if a GPU is available for computation, However, you can force CPU usage if the flag is set (good luck!)",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=1024,
        help="Number of samples contained in a mini batch (default: 1024)",
    )
    # Classification
    parser.add_argument(
        "-rrfc",
        "--rerun-frame-classification",
        action="store_true",
        help="If set, runs the frame classification after the frame correcting again (just for evaluation)",
    )
    parser.add_argument(
        '-pm',
        '--pretrained-model',
        type=str,
        choices=["ProtBert"],
        default="ProtBert",
        help="Download Pretrained Models used for inference"
    )
    return parser.parse_args()


class SequenceReadingsDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_file: str) -> None:
        """ Creates PyTorch dataset from a FASTA file and prepares it for the pipeline

        Args:
            fasta_file (str): Path to the FASTA file.
        """
        super(SequenceReadingsDataset, self).__init__()

        self.sequence_id = []
        self.dna_sequence = []
        self.aa_sequence = []
        self.contains_stop = []

        with open(fasta_file) as file:
            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    self.sequence_id.append(line[1:])
                else:
                    # Cut sequences into a length of 3
                    if len(line) % 3 != 0:
                        off_set = len(line) - len(line) % 3
                        line = line[:off_set]
                    # Have a look at the amino acid and check if it has a STOP codon in it
                    for frame in range(0, 6):
                        aa_seq = str(Seq(shift_sequence(line, frame)).translate())
                        # If no STOP codon is in it replace the original sequence
                        if "*" not in aa_seq:
                            line = shift_sequence(line, frame)
                            break                    
                    self.dna_sequence.append(line)
                    # Translate DNA and prepare string for the explorer
                    aa_seq = Seq(line).translate()
                    self.aa_sequence.append(" ".join(aa_seq))
                    # Tag if the translation got a STOP codon in it
                    if "*" in aa_seq:
                        self.contains_stop.append(True)
                    else:
                        self.contains_stop.append(False)

    def __getitem__(self, index):
        return self.aa_sequence[index]

    def __len__(self):
        return len(self.sequence_id)


class Frame_Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        super().__init__()
        self.corrected_seq = sequences

    def __getitem__(self, index):
         return self.corrected_seq[index]

    def __len__(self):
        return len(self.corrected_seq)


class PretrainedModels():
    def __init__(self, model_type: str, pretrained_model_path: str = "./.models") -> None:
        super(PretrainedModels, self).__init__()
        self.type = model_type
        self.path = os.path.join(pretrained_model_path, model_type)
        self._download_pretrained_model()
        self._init_tokonizer()

    def _init_tokonizer(self):
        if self.type == "ProtBert":
            self.tokenizer = BertTokenizer(os.path.join(self.path, "source/vocab.txt"), do_lower_case=False)

    def get_frame_classifier(self):
        if self.type == "ProtBert":
            with open(os.path.join(self.path, "frame/metadata.json")) as file:
                frame_hyperparams = json.load(file)["hparams"]

            model = ProtTransClassification(
                path=os.path.join(self.path, "source/"),
                class_num=6,
                classification_feature=frame_hyperparams["classification_feature"],
                dropout=frame_hyperparams["classification_dropout"],
            )

            model.load_state_dict(
                torch.load(
                    os.path.join(self.path, "frame/state_dict.pth"), map_location=device
                )
            )
            return model

    def get_taxonomic_classifier(self):
        if self.type == "ProtBert":
            with open(os.path.join(self.path, "taxonomic/metadata.json")) as file:
                frame_hyperparams = json.load(file)["hparams"]

            model = ProtTransClassification(
                path=os.path.join(self.path, "source/"),
                class_num=3,
                classification_feature=frame_hyperparams["classification_feature"],
                dropout=frame_hyperparams["classification_dropout"],
            )

            model.load_state_dict(
                torch.load(
                    os.path.join(self.path, "taxonomic/state_dict.pth"), map_location=device
                )
            )
            return model

    def _download_pretrained_model(self) -> None:
        """ Downloads the pretrained classification models for a certain model type.
        """
        download_urls = {
            "ProtBert": {"source": ["https://s3.amazonaws.com/models.huggingface.co/bert/Rostlab/prot_bert/config.json",
                                    "https://cdn.huggingface.co/Rostlab/prot_bert/pytorch_model.bin",
                                    "https://cdn.huggingface.co/Rostlab/prot_bert/vocab.txt"],
                         "frame": ["https://zenodo.org/record/4306420/files/metadata.json",
                                   "https://zenodo.org/record/4306420/files/state_dict.pth"],
                         "taxonomic": ["https://zenodo.org/record/4306499/files/metadata.json",
                                       "https://zenodo.org/record/4306499/files/state_dict.pth"]}
        }
        
        def reporthook(count: int, block_size: int, total_size: int) -> None:
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write("\r %d%% | %d MB | %d KB/s" % (percent, progress_size / (1024 * 1024), speed))
            sys.stdout.flush()

        if not os.path.isdir(self.path):
            url_dict = download_urls[self.type]
            for key in url_dict:
                subfolder_path = os.path.join(self.path, key)
                create_dir(subfolder_path)
                if os.listdir(subfolder_path):
                    print("There are already files in {}.".format(subfolder_path))
                else:
                    print("Downloading pre-trained models")
                    for url in url_dict[key]:
                        file_name = url.split("/")[-1]
                        urlretrieve(url, filename=os.path.join(subfolder_path, file_name), reporthook=reporthook)
                        print(" - {} successfully downloaded".format(file_name))


def shift_sequence(sequence: str, frame: int) -> str:
    """Shift and/or reverse-complement a DNA sequence into the given frame.

    Args:
        sequence (str): DNA sequence
        frame (int): The frame to move the sequence to. Possible options are:
                     - 0: Don't do anything
                     - 1, 2: Shift by 1 or 2 bases
                     - 3: Reverse-complement the sequence
                     - 4, 5: Reverse-complementary the sequence and shift by 1 or 2 bases
    Returns:
        str: Shifted DNA sequence
    """
    if frame == 0:
        return sequence
    elif frame <= 2:
        return sequence[frame:-(3-frame)]
    elif frame == 3:
        return str(Seq(sequence).reverse_complement())
    else:
        return str(Seq(sequence[(6-frame):-(frame-3)]).reverse_complement())


def create_dir(path: str) -> None:
    """Creates (sub-)folder under a given path 

    Args:
        path (str): Path under which the folder is created.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created dir: {}".format(path))


if __name__ == "__main__":
    print("Starting script...")

    # Parse configuration
    args = parse_args()

    # Set computational device
    if args.cpu is True:
        device = "cpu"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Calculations will be executed on: ", device)

    # Create output folder. User defined name or subfolder in a result directory with input file name
    if args.output is None:
        output_path = os.path.join("./results/", args.input.split(".fasta")[-2].split("/")[-1])
        create_dir(output_path)
        args.output = output_path
    else:
        create_dir(args.output)

    # Initialize pretrained models
    pretrained_models = PretrainedModels(args.pretrained_model)

    # Prepare data
    dataset = SequenceReadingsDataset(fasta_file=args.input)
    print("Dataset loaded ({} items) ".format(len(dataset)))
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=args.batch_size, collate_fn=collate_tensors)

    # Get frame classifier
    model_frame = pretrained_models.get_frame_classifier()
    model_frame.to(device)
    model_frame.eval()

    # Create result DataFrame
    results = pd.DataFrame(
        data={
            "seq_id": dataset.sequence_id,
            "dna": dataset.dna_sequence,
            "aa": dataset.aa_sequence,
            "stop_codons": dataset.contains_stop,
        }
    )

    # Predict the frames
    print("Starting frame classification:")
    pred_frame = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs = pretrained_models.tokenizer.batch_encode_plus(data, add_special_tokens=True, padding=True, truncation=True, max_length=102, return_tensors="pt")
            output = model_frame(inputs["input_ids"].to(device), inputs["token_type_ids"].to(device), inputs["attention_mask"].to(device))
            pred_frame.append(output["logits"])

    pred_frame = torch.cat(pred_frame)
    _, pred_frame_top1 = torch.max(torch.exp(pred_frame), 1)
    results["frame_pred"] = pred_frame_top1.cpu().detach()
    # May save the logits
    if args.verbose: 
        np.save(os.path.join(args.output, "data_frame-logits"), pred_frame.cpu().detach().numpy())
    
    # Copy the input and correct (shift) sequence that are of frame
    print("Starting frame correction...")
    results["aa_shifted"] = results["aa"].copy()

    mapping = {"1": 2, "2": 1, "3": 3, "4": 5, "5": 4}
    for idx, pred in enumerate(results["frame_pred"]):
        if pred != 0:
            shifted_seq = shift_sequence(results["dna"][idx], mapping[str(pred)])
            shifted_aa = Seq(shifted_seq).translate()
            results.at[idx, "aa_shifted"] = " ".join(shifted_aa)

    # Corrected dataset
    shifted_aa_set = Frame_Dataset(results["aa_shifted"])
    dataloader_frame = DataLoader(
        shifted_aa_set, num_workers=4, shuffle=False, batch_size=args.batch_size
    )

    # Get taxonomic classifier
    model_tax = pretrained_models.get_taxonomic_classifier()
    model_tax.to(device)
    model_tax.eval()

    # Taxonomix classification
    print("Starting species classification:")
    pred_species = []
    with torch.no_grad():
        for data in tqdm(dataloader_frame):
            inputs = pretrained_models.tokenizer.batch_encode_plus(data, add_special_tokens=True, padding=True, truncation=True, max_length=102, return_tensors="pt")
            output = model_tax(inputs["input_ids"].to(device), inputs["token_type_ids"].to(device), inputs["attention_mask"].to(device))
            pred_species.append(output["logits"])

    pred_species = torch.cat(pred_species)
    _, pred_species_top1 = torch.max(torch.exp(pred_species), 1)
    results["species_pred"] = pred_species_top1.cpu().detach()
    # May save the logits
    if args.verbose:
        np.save(os.path.join(args.output, "data_species-logits"), pred_species.cpu().detach().numpy())

# Write FASTA file
fasta_output_file = os.path.join(args.output, args.input.split("/")[-1])
with open(fasta_output_file, 'w') as f_out:
    for idx, row in results.iterrows():
        f_out.write(row["seq_id"] + '|' + str(row["stop_codons"]) + '|' + str(row["frame_pred"]) + '|' + str(row["species_pred"]) + "\n")
        f_out.write(row["dna"] + "\n")

# May write the pandas dataframe
if args.verbose:
    file_path = os.path.join(args.output, "data_classification-results.h5")
    results.to_hdf(file_path, key='classification_results', mode='w', format='table')

print("All results successfully stored in the folder {}".format(args.output))
print("... script finished.")
