import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from Bio.Seq import Seq


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


def frame_correction(dna_sequences, frame_predictions):
    mapping = {"0": 0, "1": 2, "2": 1, "3": 3, "4": 5, "5": 4}
    aa_sequences = []
    for idx, pred in enumerate(frame_predictions):
        shifted_seq = shift_sequence(dna_sequences[idx], mapping[str(pred)])
        shifted_aa = Seq(shifted_seq).translate()
        aa_sequences.append(" ".join(shifted_aa))
    return aa_sequences


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


def accuracy_to_stdout(conf_matrix, class_dict):
    print("Overall accuracy: {}".format(np.trace(conf_matrix)))
    for i in range(len(class_dict)):
        clmn_sum = conf_matrix[:, i].sum()
        print("In class: ", class_dict[str(i)])
        inner_class_probs = []
        for j in range(len(class_dict)):
            inner_class_probs.append(conf_matrix[j, i] / clmn_sum)
            print(" Classified as {}: {}".format(class_dict[str(j)], inner_class_probs[j]))
