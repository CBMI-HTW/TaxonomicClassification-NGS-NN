import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchnlp.utils import collate_tensors
from src.utils import create_dir, plot_confusion_heatmap, plot_roc, accuracy_to_stdout, frame_correction
from src.data import Frame_Dataset, SequenceReadingsDataset
from src.pretrained_model import PretrainedModels, predict


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
        default="./paper-results"
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


def reproduce_frame(pretrained_models: PretrainedModels, batch_size: int = 64, output_dir: str = "./results/frame"):
    print("Test Frame-Classifier...")
    create_dir(output_dir)

    model = pretrained_models.frame

    # Frame Dataset/Dataloader
    dataset = SequenceReadingsDataset(test_type="frame")
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, collate_fn=collate_tensors)

    # Create result DataFrame
    results = pd.DataFrame(data={"seq_id": dataset.sequence_id,
                                 "dna": dataset.dna_sequence,
                                 "aa": dataset.aa_sequence,
                                 "stop_codons": dataset.contains_stop,
                                 "frame": dataset.label_frame})

    # Frame classification
    logits, results["frame_pred"] = predict(model, dataloader, pretrained_models.tokenizer, pretrained_models.device)
    np.save(os.path.join(output_dir, "frame_logits"), logits)

    # Confusion matrix
    plot_confusion_heatmap(results["frame"], results["frame_pred"], os.path.join(output_dir, "frame_conf-matrix.png"), ["Actual Frame"], ["Frame Prediction"], normalize=True)

    # ROC
    plot_roc(logits, results["frame"].to_numpy(), save_fig=os.path.join(output_dir, "frame_ROC.png"))

    # Accuracy
    confusion_matrix = pd.crosstab(results["frame"], results["frame_pred"], normalize=True).to_numpy()
    frame_classes = {"0": "on-frame", "1": "offset by one base", "2": "offset by two bases", "3": "reverse-complementary", "4": "reverse-complementary and offset by one base", "5": "reverse complementary and offset by two bases"}
    accuracy_to_stdout(confusion_matrix, frame_classes)

    # Correct frames
    results["aa_shifted"] = frame_correction(results["dna"], results["frame_pred"])
    dataloader = DataLoader(Frame_Dataset(results["aa_shifted"]), num_workers=4, shuffle=False, batch_size=batch_size)

    # Rerun frame xlassification
    _, results["aa_shifted_frame_pred"] = predict(model, dataloader, pretrained_models.tokenizer, pretrained_models.device)

    # Evaluate Frame Re-Classification
    plot_confusion_heatmap(results["frame"], results["aa_shifted_frame_pred"], os.path.join(output_dir, "shifted_frame_conf-matrix.png"), ["Actual Frame"], ["Frame Prediction"], normalize=True)

    # Save results and checkout
    results.to_hdf(os.path.join(output_dir, "results_dataframe.h5"), key='classification_results', mode='w', format='table')
    print("... finished. Results are saved in {}\n".format(output_dir))


def reproduce_taxonomic_classifier_testset(pretrained_models: PretrainedModels, batch_size: int = 64, output_dir: str = "./results/taxonomic") -> None:
    print("Test Taxonomic-Classifier...")
    create_dir(output_dir)

    # Get Taxonomix testset
    dataset = SequenceReadingsDataset(test_type="taxonomic")
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size)

    # Create Result DataFrame for Taxonomic Classification
    results = pd.DataFrame(data={"seq_id": dataset.sequence_id,
                                 "aa": dataset.aa_sequence,
                                 "stop_codons": dataset.contains_stop,
                                 "species": dataset.label_species})

    # Taxonomic Testset Classification
    logits, results["tax_pred"] = predict(pretrained_models.taxonomic, dataloader, pretrained_models.tokenizer, pretrained_models.device)
    np.save(os.path.join(output_dir, "logits"), logits)

    # Confusion Matrix
    plot_confusion_heatmap(results["species"], results["tax_pred"], os.path.join(output_dir, "taxonomic_conf-matrix.png"), ["Actual Class"], ["Prediction"], normalize=True)   

    # ROC
    plot_roc(logits, results["species"].to_numpy(), save_fig=os.path.join(output_dir, "taxonomic_ROC.png"))

    # Accuracy
    confusion_matrix = pd.crosstab(results["species"], results["tax_pred"], normalize=True).to_numpy()
    accuracy_to_stdout(confusion_matrix, {"1": "Bacteria", "0": "Virus", "2": "Human"})

    # Save results and checkout
    results.to_hdf(os.path.join(output_dir, "result_dataframe.h5"), key='classification_results', mode='w', format='table')
    print("... finished. Results are saved in {}\n".format(output_dir))


def reproduce_inORF(pretrained_models: PretrainedModels, batch_size: int = 64, output_dir: str = "./results/inORF") -> None:
    print("Experiment with inORF dataset...")
    create_dir(output_dir)

    # Get dataset
    dataset = SequenceReadingsDataset(test_type="inORF")
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size)

    # Create Result DataFrame for Taxonomic Classification
    results = pd.DataFrame(data={"seq_id": dataset.sequence_id,
                                 "dna": dataset.dna_sequence,
                                 "aa": dataset.aa_sequence,
                                 "stop_codons": dataset.contains_stop,
                                 "species": dataset.label_species})

    # Frame classification
    logits, results["frame_pred"] = predict(pretrained_models.frame, dataloader, pretrained_models.tokenizer, pretrained_models.device)
    np.save(os.path.join(output_dir, "frame_logits"), logits)

    # Correct Frames
    results["aa_shifted"] = frame_correction(results["dna"], results["frame_pred"])
    dataloader = DataLoader(Frame_Dataset(results["aa_shifted"]), num_workers=4, shuffle=False, batch_size=batch_size)

    # Taxonomic classification
    logits, results["tax_pred"] = predict(pretrained_models.taxonomic, dataloader, pretrained_models.tokenizer, pretrained_models.device)
    np.save(os.path.join(output_dir, "taxonomic_logits"), logits)

    # Confusion matrix
    plot_confusion_heatmap(results["species"], results["tax_pred"], os.path.join(output_dir, "taxonomic_conf-matrix.png"), ["Actual Class"], ["Prediction"], normalize=True)

    # ROC
    plot_roc(logits, results["species"].to_numpy(), save_fig=os.path.join(output_dir, "taxonomic_ROC.png"))

    # Accuracy
    confusion_matrix = pd.crosstab(results["species"], results["tax_pred"], normalize=True).to_numpy()
    accuracy_to_stdout(confusion_matrix, {"1": "Bacteria", "0": "Virus", "2": "Human"})

    # Save results and checkout
    results.to_hdf(os.path.join(output_dir, "result_dataframe.h5"), key='classification_results', mode='w', format='table')
    print("... finished. Results are saved in {}\n".format(output_dir))


def reproduce_SRR(pretrained_models: PretrainedModels, batch_size: int = 64, output_dir: str = "./results/SRR") -> None:
    print("Experiment with SRR dataset ...")
    create_dir(output_dir)

    # Get dataset
    dataset = SequenceReadingsDataset(test_type="SRR")
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size)

    # Create Result DataFrame for Taxonomic Classification
    results = pd.DataFrame(data={"seq_id": dataset.sequence_id,
                                 "dna": dataset.dna_sequence,
                                 "aa": dataset.aa_sequence,
                                 "stop_codons": dataset.contains_stop,
                                 "species": dataset.label_species})

    # Frame classification
    logits, results["frame_pred"] = predict(pretrained_models.frame, dataloader, pretrained_models.tokenizer, pretrained_models.device)
    np.save(os.path.join(output_dir, "frame_logits"), logits)

    # Correct Frames
    results["aa_shifted"] = frame_correction(results["dna"], results["frame_pred"])
    dataloader = DataLoader(Frame_Dataset(results["aa_shifted"]), num_workers=4, shuffle=False, batch_size=batch_size)

    # Taxonomix classification
    logits, results["tax_pred"] = predict(pretrained_models.taxonomic, dataloader, pretrained_models.tokenizer, pretrained_models.device)
    np.save(os.path.join(output_dir, "taxonomic_logits"), logits)

    # Confusion matrix
    plot_confusion_heatmap(results["species"], results["tax_pred"], os.path.join(output_dir, "taxonomic_conf-matrix.png"), ["Actual Class"], ["Prediction"], normalize=True)

    # ROC
    plot_roc(logits, results["species"].to_numpy(), save_fig=os.path.join(output_dir, "taxonomic_ROC.png"))

    # Accuracy
    confusion_matrix = pd.crosstab(results["species"], results["tax_pred"], normalize=True).to_numpy()
    accuracy_to_stdout(confusion_matrix, {"1": "Bacteria", "0": "Virus", "2": "Human"})

    # Save results and checkout
    results.to_hdf(os.path.join(output_dir, "result_dataframe.h5"), key='classification_results', mode='w', format='table')
    print("... finished. Results are saved in {}\n".format(output_dir))


if __name__ == "__main__":
    print("Starting paper result reproduction...")
    # Argparser and some settings
    args = parse_args()
    sns.set_theme()
    sns.color_palette()
    sns.set_context("paper")

    # Check if a GPU is available
    if args.cpu is True:
        device = "cpu"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Calculations will be executed on: ", device)

    # Initialize pretrained models
    print("Initialize pre-trained models...")
    pretrained_models = PretrainedModels("ProtBert", device=device)
    print("... done.")

    # Frame Classification
    reproduce_frame(pretrained_models, args.batch_size, os.path.join(args.output, "frame"))

    # Taxonomic Classification
    reproduce_taxonomic_classifier_testset(pretrained_models, args.batch_size, os.path.join(args.output, "taxonomic"))    

    # inORF Classification
    reproduce_inORF(pretrained_models, args.batch_size, os.path.join(args.output, "inORF"))

    # SRR Classification
    reproduce_SRR(pretrained_models, args.batch_size, os.path.join(args.output, "SRR"))
    print("... script finished.")
