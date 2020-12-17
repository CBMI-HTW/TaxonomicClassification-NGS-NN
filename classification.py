
import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchnlp.utils import collate_tensors
from src.data import Frame_Dataset, SequenceReadingsDataset
from src.pretrained_model import PretrainedModels, predict
from src.utils import create_dir, frame_correction


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
        '-pm',
        '--pretrained-model',
        type=str,
        choices=["ProtBert"],
        default="ProtBert",
        help="Download Pretrained Models used for inference"
    )
    return parser.parse_args()


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
    pretrained_models = PretrainedModels(args.pretrained_model, device=device)

    # Prepare data
    dataset = SequenceReadingsDataset(args.input)
    print("Dataset loaded ({} items) ".format(len(dataset)))
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=args.batch_size, collate_fn=collate_tensors)

    # Create result DataFrame
    results = pd.DataFrame(data={"seq_id": dataset.sequence_id,
                                 "dna": dataset.dna_sequence,
                                 "aa": dataset.aa_sequence,
                                 "stop_codons": dataset.contains_stop})

    # Frame classification
    print("Starting frame classification:")
    logits, results["frame_pred"] = predict(pretrained_models.frame, dataloader, pretrained_models.tokenizer, pretrained_models.device)
    if args.verbose:
        np.save(os.path.join(args.output, "frame_logits"), logits)

    # Correct dataset
    results["aa_shifted"] = frame_correction(results["dna"], results["frame_pred"])
    dataloader = DataLoader(Frame_Dataset(results["aa_shifted"]), num_workers=4, shuffle=False, batch_size=args.batch_size)

    # Taxonomix classification
    print("Starting species classification:")
    logits, results["species_pred"] = predict(pretrained_models.taxonomic, dataloader, pretrained_models.tokenizer, pretrained_models.device)
    if args.verbose:
        np.save(os.path.join(args.output, "taxonomix_logits"), logits)

# Write FASTA file
fasta_output_file = os.path.join(args.output, args.input.split("/")[-1])
with open(fasta_output_file, 'w') as f_out:
    for idx, row in results.iterrows():
        f_out.write(row["seq_id"] + '|' + str(row["stop_codons"]) + '|' + str(row["frame_pred"]) + '|' + str(row["species_pred"]) + "\n")
        f_out.write(row["dna"] + "\n")

# May write the pandas dataframe
if args.verbose:
    file_path = os.path.join(args.output, "result_dataframe.h5")
    results.to_hdf(file_path, key='classification_results', mode='w', format='table')

print("All results successfully stored in the folder {}".format(args.output))
print("... script finished.")
