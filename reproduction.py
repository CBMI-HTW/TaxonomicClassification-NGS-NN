
import os
import sys
import argparse
import time
import json
import shutil
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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def parse_args():
    parser = argparse.ArgumentParser(
        description="Taxonomic Classification of NGS Reads using Language Models"
    )
    # General
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path where the evaluation results are stored",
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
    return parser.parse_args()


class PaperTestSets(torch.utils.data.Dataset):
    def __init__(self, type) -> None:
        """ Creates PyTorch dataset from a FASTA file and prepares it for the pipeline

        Args:
            fasta_file (str): Path to the FASTA file.
        """
        super().__init__()

        self.sequence_id = []
        self.aa_sequence = []
        self.contains_stop = []
        self.label_frame = []
        self.dna_sequence = []
        self.label_species = []

        # Check if test files exist, if not download
        if not (os.path.isfile("./data/frame/refseq_ds_all_off-frames_fb_DNA_test.fasta")  and os.path.isfile("./data/taxonomic/uniprot_swiss-prot_vbh_p100d_w_test.fasta")):
            self._download_testsets()

        if type == "frame":
            fasta_file = "./data/frame/refseq_ds_all_off-frames_fb_DNA_test.fasta"
        elif type == "taxonomic":
            fasta_file = "./data/taxonomic/uniprot_swiss-prot_vbh_p100d_w_test.fasta"
        else:
            print("Illegal type value")

        with open(fasta_file) as file:
            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    self.sequence_id.append(line.split("|")[0])
                    if type == "frame":
                        self.label_frame.append(int(line[-1]))
                    else:
                        self.label_species.append(int(line[-1]))
                else:
                    # Cut sequences into a length of 3
                    if len(line) % 3 != 0:
                        off_set = len(line) - len(line) % 3
                        line = line[:off_set]
                    if type == "frame":
                    # Have a look at the amino acid and check if it has a STOP codon in it
                        for frame in range(0, 6):
                            aa_seq = str(Seq(shift_sequence(line, frame)).translate())
                            # If no STOP codon is in it replace the original sequence
                            if "*" not in aa_seq:
                                line = shift_sequence(line, frame)
                                break                    
                        self.dna_sequence.append(line)
                        line = Seq(line).translate()
                    # Translate DNA and prepare string for the explorer

                    self.aa_sequence.append(" ".join(line))
                    # Tag if the translation got a STOP codon in it
                    if "*" in line:
                        self.contains_stop.append(True)
                    else:
                        self.contains_stop.append(False)

    def _download_testsets(self):
        """ Downloads the pretrained classification models for a certain model type.
        """
        download_urls = {
            "frame": "https://redmine.f4.htw-berlin.de/owncloud/index.php/s/JaWXZbosBaB6r4W/download",
            "taxonomic": "https://redmine.f4.htw-berlin.de/owncloud/index.php/s/RTeykYdPijsGL2w/download"
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

        for key in download_urls:
            subfolder_path = os.path.join("./data", key)
            create_dir(subfolder_path)
            # Download
            file_name = download_urls[key].split("/")[-1]
            file_path = os.path.join(subfolder_path, file_name)
            urlretrieve(download_urls[key], filename=file_path, reporthook=reporthook)
            print(" - {} successfully downloaded".format(file_name))
            # Unzip
            shutil.unpack_archive(file_path, extract_dir=subfolder_path, format="gztar")
            # Remove downloaded archive
            os.remove(file_path)

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


def plot_confusion_heatmap(label, preds, save_path, x_name=None, y_name=None, normalize=False):
    confusion_matrix = pd.crosstab(label, preds, rownames=x_name, colnames=y_name, normalize=normalize)
    sns.heatmap(confusion_matrix, annot=True, vmin=0, vmax=1, cmap="Blues")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc(predictions, targets, avg_only=False, title="ROC", save_fig=None):
    """ Creates a receiver operating characteristic plot for multiclass predictions

    Compute ROC curve and ROC area for each class - heavily based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Args:
    predictions: Numpy array
        Array represents the predictions of a neural network
    targets: Numpy array
        Ground truth in vector representation
    avg_only: Bool - default: False

    verbose: Bool - default: False
        If 'True' in addition the confusion matrix for the prediction is printed
    save_fig: String - default: None
        If a path is given, the plot will be saved in that location.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Extract number of classes
    num_classes = predictions.shape[1]

    # Compute ROC curves and AUCs
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets, predictions[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Add micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(np.eye(num_classes)[targets].ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Add macro-average:
    #  Aggregate all false positive rates -> interpolate all ROC curves at this points -> average everything and compute AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=[6, 6])

    plt.plot(fpr["micro"], tpr["micro"], label='Micro Average (Area = {0:0.2f})'.format(roc_auc["micro"]), linestyle=':', linewidth=3)
    plt.plot(fpr["macro"], tpr["macro"], label='Macro Average (Area = {0:0.2f})'.format(roc_auc["macro"]), linestyle=':', linewidth=3)

    if not avg_only:
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label='Class {0} (Area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    if save_fig is not None:
        plt.savefig(save_fig, bbox_inches="tight")
    plt.close()


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
        args.output = "./results/paper"
    create_dir(args.output)

    # Initialize pretrained models
    pretrained_models = PretrainedModels("ProtBert")

    # Prepare data
    dataset_frame = PaperTestSets("frame")
    print("Dataset loaded ({} items) ".format(len(dataset_frame)))
    dataloader_frame = DataLoader(dataset_frame, num_workers=4, shuffle=False, batch_size=args.batch_size, collate_fn=collate_tensors)

#     # Get frame classifier
    model_frame = pretrained_models.get_frame_classifier()
    model_frame.to(device)
    model_frame.eval()

    # Create result DataFrame
    results_frame = pd.DataFrame(
        data={
            "seq_id": dataset_frame.sequence_id,
            "dna": dataset_frame.dna_sequence,
            "aa": dataset_frame.aa_sequence,
            "stop_codons": dataset_frame.contains_stop,
            "frame": dataset_frame.label_frame            
        }
    )

    # Predict the frames
    print("Starting frame classification:")
    pred_frame = []
    with torch.no_grad():
        for data in tqdm(dataloader_frame):
            inputs = pretrained_models.tokenizer.batch_encode_plus(data, add_special_tokens=True, padding=True, truncation=True, max_length=102, return_tensors="pt")
            output = model_frame(inputs["input_ids"].to(device), inputs["token_type_ids"].to(device), inputs["attention_mask"].to(device))
            pred_frame.append(output["logits"])

    pred_frame = torch.cat(pred_frame)
    _, pred_frame_top1 = torch.max(torch.exp(pred_frame), 1)
    results_frame["frame_pred"] = pred_frame_top1.cpu().detach()
    # May save the logits
    np.save(os.path.join(args.output, "data_frame-logits"), pred_frame.cpu().detach().numpy())
    
    # Copy the input and correct (shift) sequence that are of frame
    print("Starting frame correction...")
    results_frame["aa_shifted"] = results_frame["aa"].copy()
    mapping = {"1": 2, "2": 1, "3": 3, "4": 5, "5": 4}
    for idx, pred in enumerate(results_frame["frame_pred"]):
        if pred != 0:
            shifted_seq = shift_sequence(results_frame["dna"][idx], mapping[str(pred)])
            shifted_aa = Seq(shifted_seq).translate()
            results_frame.at[idx, "aa_shifted"] = " ".join(shifted_aa)
    print("... finished")

    dataloader_frame_cor = torch.utils.data.DataLoader(Frame_Dataset(results_frame["aa_shifted"]), num_workers=4, shuffle=False, batch_size=args.batch_size)

    # ReRun the Frame Classification to see if we have an improvement
    print("Starting frame re-classification:")
    pred_corrected_frame = []
    with torch.no_grad():
        for data in tqdm(dataloader_frame_cor):
            inputs = pretrained_models.tokenizer.batch_encode_plus(
                data,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=102,
                return_tensors="pt",
            )

            output = model_frame(inputs["input_ids"].to(device), inputs["token_type_ids"].to(device), inputs["attention_mask"].to(device))
            _, frame_shift_pred = torch.max(torch.exp(output["logits"]), 1)
            pred_corrected_frame.append(frame_shift_pred)
    pred_corrected_frame = torch.cat(pred_corrected_frame)
    results_frame["aa_shifted_frame_pred"] = pred_corrected_frame.cpu().detach()

    # Create corrected dataset
    dataset_taxonomic = PaperTestSets("taxonomic")
    dataloader_frame = DataLoader(dataset_taxonomic, num_workers=4, shuffle=False, batch_size=args.batch_size)

    # Create result DataFrame
    results_tax = pd.DataFrame(
        data={
            "seq_id": dataset_taxonomic.sequence_id,
            "aa": dataset_taxonomic.aa_sequence,
            "stop_codons": dataset_taxonomic.contains_stop,
            "species": dataset_taxonomic.label_species            
        }
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
    results_tax["species_pred"] = pred_species_top1.cpu().detach()
    # May save the logits
    np.save(os.path.join(args.output, "data_species-logits"), pred_species.cpu().detach().numpy())


    ##############
    # Evaluation #
    ##############
    # Confusion matrix
    plot_confusion_heatmap(results_frame["frame"], results_frame["frame_pred"], os.path.join(args.output, "frame_confusion_matrix.png"), ["Actual Frame"], ["Frame Prediction"], normalize=True)
    plot_confusion_heatmap(results_frame["frame"], results_frame["aa_shifted_frame_pred"], os.path.join(args.output, "frame_shifted_confusion_matrix.png"), [" "], ["Frame Prediction"], normalize=True)
    plot_confusion_heatmap(results_tax["species"], results_tax["species_pred"], os.path.join(args.output, "species_confusion_matrix.png"), ["Actual Class"], ["Prediction"], normalize=True)

    # ROC
    plot_roc(pred_frame.cpu().detach().numpy(), results_frame["frame"].to_numpy(), save_fig=os.path.join(args.output, "frame_roc.png"))
    plot_roc(pred_species.cpu().detach().numpy(), results_tax["species"].to_numpy(), save_fig=os.path.join(args.output, "species_roc.png"))

    # Accuracy
    confusion_matrix = pd.crosstab(results_frame["frame"], results_frame["frame_pred"], normalize=True).to_numpy()
    print("Total Frame Accuracy: ", np.trace(confusion_matrix))
    for i in range(6):
        clmn_sum = confusion_matrix[:, i].sum()
        print("Class", str(i))
        for j in range(6):
            print(" Classified as {}: {}".format(str(j), confusion_matrix[i, j] / clmn_sum))
    print(30*"-")
    confusion_matrix = pd.crosstab(results_tax["species"], results_tax["species_pred"], normalize=True).to_numpy()
    print("Total Sepcies Accuracy: ", np.trace(confusion_matrix))
    tax = {"1": "Bacteria", "0": "Virus", "2": "Human"}
    for i in range(3):
        clmn_sum = confusion_matrix[:, i].sum()
        print("Class", tax[str(i)])
        inner_class_probs = []
        for j in range(3):
            inner_class_probs.append(confusion_matrix[j, i] / clmn_sum)
            print(" Classified as {}: {}".format(tax[str(j)], inner_class_probs[j]))
        print(np.array(inner_class_probs).sum())


print("All results successfully stored in the folder {}".format(args.output))
print("... script finished.")
