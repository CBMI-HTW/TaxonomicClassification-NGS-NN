import torch
import torch.nn as nn
from transformers import BertModel


class ProtTransClassification(nn.Module):
    def __init__(self, path: str, class_num: int, classification_feature: int = 512, dropout: float = 0.5, freeze_bert: bool = False) -> None:
        """ Creates a custom BERT model with a classification head.
            Base is a pretrained BERT model (https://github.com/huggingface/transformers, https://github.com/agemagician/ProtTrans)
            with an additional classification layer.

        Args:
            path (str): Path to folder(!) for necessary BERT model files (pytorch_model.bin, vocab.txt, config.json)
            class_num (int): Number of classes
            classification_feature (int, optional): Base number to create features in the classification layers. Defaults to 1024.
        """
        super(ProtTransClassification, self).__init__()

        # Set class attributs
        self.class_num = class_num
        self.classification_feature = classification_feature
        self.dropout = dropout

        # Load pretrained BERT model
        self.wordencoding = BertModel.from_pretrained(path)

        if freeze_bert is True:
            for param in self.wordencoding.parameters():
                param.requires_grad = False

        # Classification layer
        self.classification = nn.Sequential(
            nn.Linear(1024, self.classification_feature),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.classification_feature, self.class_num),
            nn.LogSoftmax(dim=1)
        )
    # adapted from https://github.com/UKPLab/sentence-transformers/blob/eb39d0199508149b9d32c1677ee9953a84757ae4/sentence_transformers/models/Pooling.py
    def pool_strategy(self, features: dict, pool_cls: bool = True, pool_max: bool = True, pool_mean: bool = True, pool_mean_sqrt: bool = True) -> torch.Tensor:
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        # Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_max:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt or pool_sum:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        
        output_vector = torch.cat(output_vectors, 1)

        return output_vector

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """ Forwards a batch through a pretrained BERT model and classifies the resulting
            word embeddings with a fully connected neural network.

        Args:
            input_ids (torch.Tensor): Input sequence tensor
            token_type_ids (torch.Tensor): Tensor of 0,1 indicating if a token belongs to seq0 or seq1
            attention_mask (torch.Tensor): Tensor of 0,1 indicating if a token is masked (0) or not (1)

        Returns:
            dict: {"logits": torch.Tensor} - Classification logits for each input
        """
        # input_ids = torch.tensor(input_ids, device=self.device)
        # attention_mask = torch.tensor(attention_mask,device=self.device)

        # Forward the batch through the pretrained BERT model
        word_embeddings = self.wordencoding(input_ids, attention_mask)[0]

        # Prepare the word embeddings, e.i. create one vector for a classification 
        pooling = self.pool_strategy({"token_embeddings": word_embeddings,
                                      "cls_token_embeddings": word_embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      }, pool_cls=False, pool_max=False, pool_mean=True, pool_mean_sqrt=False)

        # Small FC-NN as classification head
        logits = self.classification(pooling)

        return {"logits": logits}
        