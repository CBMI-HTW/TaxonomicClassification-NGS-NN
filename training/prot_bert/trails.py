import torch
from torch.utils.data import RandomSampler

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from determined.tensorboard.metric_writers.pytorch import TorchWriter

from transformers import BertTokenizer

from model import ProtTransClassification
from utils import Lamb, UniProtData, accuracy


class SpeciesClassification(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        # Read configuration
        self.context = context
        self.data_config = self.context.get_data_config()

        # Create Tensorboard logger
        self.logger = TorchWriter()

        # Create tokinizer based on the predefient vocabulary
        self.tokenizer = BertTokenizer(self.data_config["voc_path"], do_lower_case=False)

        # Initialize model and wrap it in the determined api
        model = ProtTransClassification(self.data_config["pretrained_path"],
                                        class_num=3,
                                        classification_feature=self.context.get_hparam("classification_feature"),
                                        dropout=self.context.get_hparam("classification_dropout"),
                                        freeze_bert=self.context.get_hparam("bert_freeze"))

        optimizer = Lamb([{"params": model.wordencoding.parameters(), "lr": self.context.get_hparam("bert_lr")},
                          {"params": model.classification.parameters()}], lr=self.context.get_hparam("classification_lr"))

        self.model = self.context.wrap_model(model)
        self.optimizer = self.context.wrap_optimizer(optimizer)

    def build_training_data_loader(self) -> DataLoader:
        uni_prot = UniProtData(self.tokenizer, self.context.get_hparam("sequence_length"))
        train_data = uni_prot.from_file(self.data_config["train_data"])

        data_loader = DataLoader(train_data,
                                 batch_size=self.context.get_per_slot_batch_size(),
                                 sampler=RandomSampler(train_data),
                                 collate_fn=uni_prot.prepare_sample,
                                 num_workers=self.data_config["worker"])

        return data_loader

    def build_validation_data_loader(self) -> DataLoader:
        uni_prot = UniProtData(self.tokenizer, self.context.get_hparam("sequence_length"))        
        val_data = uni_prot.from_file(self.data_config["val_data"])

        data_loader = DataLoader(val_data,
                                 batch_size=self.context.get_per_slot_batch_size(),
                                 collate_fn=uni_prot.prepare_sample,
                                 num_workers=self.data_config["worker"])

        return data_loader

    def train_batch(self, batch: tuple, epoch_idx: int, batch_idx: int) -> dict:
        """ Train pipeline for a batch

        Args:
            batch (tuple): ({"input_ids": [[]],
                             "token_type_ids": [[]],
                             "attention_mask":[[]]},
                            {"labels": [labels]})
            epoch_idx (int): Epoch number that is currently being calculated
            batch_idx (int): Batch number that is currently being calculated

        Returns:
            dict: {"loss": torch.Tensor,
                   "train_acc": torch.Tensor}
        """
        data, labels = batch

        # Run the forward path and calculate loss
        output = self.model(**data)
        loss = torch.nn.functional.nll_loss(output["logits"], labels["labels"])

        # Define the training backward pass and step the optimizer.
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        # Calculate batch acc
        train_batch_acc = accuracy(output["logits"], labels["labels"])[0]

        return {"loss": loss.item(), "train_batch_acc": train_batch_acc.item()}

    def evaluate_batch(self, batch: tuple) -> dict:
        """ Evaluation pipeline for a batch

        Args:
            batch (tuple): ({"input_ids": [[]],
                             "token_type_ids": [[]],
                             "attention_mask":[[]]},
                            {"labels": [labels]})

        Returns:
            dict: {"val_loss": torch.Tensor,
                   "val_acc": torch.Tensor}
        """
        data, labels = batch

        # Run forward path and calculate loss
        output = self.model(**data)
        val_loss = torch.nn.functional.nll_loss(output["logits"], labels["labels"]).item()

        # Get top1 and calculate accuracy
        output = torch.exp(output["logits"])
        val_acc = accuracy(output, labels["labels"])[0].item()

        return {"val_loss": val_loss, "val_acc": val_acc}


class FrameClassification(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        # Read configuration
        self.context = context
        self.data_config = self.context.get_data_config()

        # Create Tensorboard logger
        self.logger = TorchWriter()

        # Create tokinizer based on the predefient vocabulary
        self.tokenizer = BertTokenizer(self.data_config["voc_path"], do_lower_case=False)

        # Label Encoder
        if self.context.get_hparam("reduce_to_binary_problem"):
            class_num = 2
        else:
            class_num = 6

        # Initialize model and wrap it in the determined api
        model = ProtTransClassification(self.data_config["pretrained_path"],
                                        class_num=class_num,
                                        classification_feature=self.context.get_hparam("classification_feature"),
                                        dropout=self.context.get_hparam("classification_dropout"),
                                        freeze_bert=self.context.get_hparam("bert_freeze"))

        optimizer = Lamb([{"params": model.wordencoding.parameters(), "lr": self.context.get_hparam("bert_lr")},
                          {"params": model.classification.parameters()}], lr=self.context.get_hparam("classification_lr"))

        self.model = self.context.wrap_model(model)
        self.optimizer = self.context.wrap_optimizer(optimizer)

    def build_training_data_loader(self) -> DataLoader:
        uni_prot = UniProtData(self.tokenizer, self.context.get_hparam("sequence_length"))
        train_data = uni_prot.from_file(self.data_config["train_data"], reduce_to_binary=self.context.get_hparam("reduce_to_binary_problem"))

        data_loader = DataLoader(train_data, 
                                 batch_size=self.context.get_per_slot_batch_size(),
                                 sampler=RandomSampler(train_data),
                                 collate_fn=uni_prot.prepare_sample,
                                 num_workers=self.data_config["worker"])

        return data_loader

    def build_validation_data_loader(self) -> DataLoader:
        uni_prot = UniProtData(self.tokenizer, self.context.get_hparam("sequence_length"))        
        val_data = uni_prot.from_file(self.data_config["val_data"], reduce_to_binary=self.context.get_hparam("reduce_to_binary_problem"))

        data_loader = DataLoader(val_data,
                                 batch_size=self.context.get_per_slot_batch_size(),
                                 collate_fn=uni_prot.prepare_sample,
                                 num_workers=self.data_config["worker"])

        return data_loader

    def train_batch(self, batch: tuple, epoch_idx: int, batch_idx: int) -> dict:
        """ Train pipeline for a batch

        Args:
            batch (tuple): ({"input_ids": [[]],
                             "token_type_ids": [[]],
                             "attention_mask":[[]]},
                            {"labels": [labels]})
            epoch_idx (int): Epoch number that is currently being calculated
            batch_idx (int): Batch number that is currently being calculated

        Returns:
            dict: {"loss": torch.Tensor,
                   "train_acc": torch.Tensor}
        """
        data, labels = batch

        # Run the forward path and calculate loss
        output = self.model(**data)
        loss = torch.nn.functional.nll_loss(output["logits"], labels["labels"])

        # Define the training backward pass and step the optimizer.
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        # Calculate batch acc
        # train_batch_acc = accuracy(output["logits"], labels["labels"])[0]

        return {"loss": loss.item()}

    def evaluate_batch(self, batch: tuple) -> dict:
        """ Evaluation pipeline for a batch

        Args:
            batch (tuple): ({"input_ids": [[]],
                             "token_type_ids": [[]],
                             "attention_mask":[[]]},
                            {"labels": [labels]})

        Returns:
            dict: {"val_loss": torch.Tensor,
                   "val_acc": torch.Tensor}
        """
        data, labels = batch

        # Run forward path and calculate loss
        output = self.model(**data)
        val_loss = torch.nn.functional.nll_loss(output["logits"], labels["labels"]).item()

        # Get top1 and calculate accuracy
        output = torch.exp(output["logits"])
        val_acc = accuracy(output, labels["labels"])[0].item()

        return {"val_loss": val_loss, "val_acc": val_acc}


if __name__ == "__main__":
    print("go home")
