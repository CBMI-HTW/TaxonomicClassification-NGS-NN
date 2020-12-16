import sys
import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from urllib.request import urlretrieve
from tqdm import tqdm
from training.prot_bert.model import ProtTransClassification
from src.utils import create_dir


class PretrainedModels():
    def __init__(self, model_type: str, pretrained_model_path: str = "./.models", device="cpu") -> None:
        super().__init__()
        self.device = device
        self.type = model_type
        self.path = os.path.join(pretrained_model_path, model_type)
        self._download_pretrained_model()
        self._init_tokonizer()
        self._init_model(classifier="frame", class_num=6)
        self._init_model(classifier="taxonomic", class_num=3)

    def _init_tokonizer(self):
        if self.type == "ProtBert":
            self.tokenizer = BertTokenizer(os.path.join(self.path, "source/vocab.txt"), do_lower_case=False)

    def _init_model(self, classifier: str, class_num: int) -> None:
        if self.type == "ProtBert":
            with open(os.path.join(self.path, classifier, "metadata.json")) as file:
                frame_hyperparams = json.load(file)["hparams"]

            model = ProtTransClassification(
                path=os.path.join(self.path, "source/"),
                class_num=class_num,
                classification_feature=frame_hyperparams["classification_feature"],
                dropout=frame_hyperparams["classification_dropout"],
            )

            model.load_state_dict(
                torch.load(
                    os.path.join(self.path, classifier, "state_dict.pth"), map_location=self.device
                )
            )

        if classifier == "frame":
            self.frame = model.to(self.device).eval()
        elif classifier == "taxonomic":
            self.taxonomic = model.to(self.device).eval()

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


def predict(model: ProtTransClassification, dataloader: DataLoader, tokenizer: BertTokenizer, device) -> (np.array, np.array):
    logits = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs = tokenizer.batch_encode_plus(data, add_special_tokens=True, padding=True, truncation=True, max_length=102, return_tensors="pt")
            output = model(inputs["input_ids"].to(device), inputs["token_type_ids"].to(device), inputs["attention_mask"].to(device))
            logits.append(output["logits"])
    logits = torch.cat(logits)
    _, preds = torch.max(torch.exp(logits), 1)
    # Detach and convert to numpy
    logits = logits.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()
    return logits, preds
