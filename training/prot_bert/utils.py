import numpy as np

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler

from torchnlp.utils import collate_tensors
from torchnlp.encoders import LabelEncoder


class UniProtData():
    def __init__(self, tokenizer, sequence_length):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def from_file(self, fasta_file, reduce_to_binary=False):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq(reduce_to_binary):
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            if reduce_to_binary:
                if cur_seq_label == "0":
                    sequence_labels.append(cur_seq_label)
                else:
                    sequence_labels.append("1")
            else:
                sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq(reduce_to_binary)
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line.split("|")[-1]
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(" ".join(line.strip()))

        _flush_current_seq(reduce_to_binary)

        # More usefull is to check if we have equal number of sequences and labels
        assert len(sequence_strs) == len(sequence_labels)

        # Create label encoder from unique strings in the label list
        self.label_encoder = LabelEncoder(np.unique(sequence_labels), reserved_labels=[])

        self.data = []
        for i in range(len(sequence_strs)):
            self.data.append({"seq": str(sequence_strs[i]), "label": str(sequence_labels[i])})

        return self.data

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)

        # Tokenize the input, return dict with 3 entries:
        #   input_ids: tokenized matrix
        #   token_input_id: matrix of 0,1 indicating if the element belongs to seq0 or eq1
        #   attention_mask: matrix of 0,1 indicating if a token ist masked (0) or not (1)
        # Convert to PT tensor
        inputs = self.tokenizer.batch_encode_plus(sample["seq"],
                                                  add_special_tokens=True,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.sequence_length,
                                                  return_tensors="pt")

        if prepare_target is False:
            return inputs, {}

        # Prepare target:
        try:
            targets = {"labels": self.label_encoder.batch_encode(sample["label"])}
            return inputs, targets
        except RuntimeError:
            print(sample["label"])
            raise Exception("Label encoder found an unknown label.")

    def get_dataloader(self, file_path, batch_size, num_worker=4):
        data = self.from_file(file_path)

        data_loader = DataLoader(data, batch_size=batch_size, sampler=RandomSampler(data), collate_fn=self.prepare_sample, num_workers=num_worker)

        return data_loader

# https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
class Lamb(Optimizer):
    """Implements Lamb algorithm. It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> float:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Simple debug function to check which parameters are actualy learned
def print_learnable_params(model, freezed_too=False, log_file=None) -> None:
    """ Print (learnable) parameters in a given network model

    Args:
        model: Model you want to print the learned parameters
        freezed_too: Print the freezed parameters as well
        log_file: if a opened log file is provided, information is written there]
    """
    updated = []
    freezed = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            updated.append(name)
        else:
            freezed.append(name)

    print("\nFollowing parameters of the model will be updated:", file=log_file)

    for para in updated:
        print("- {}".format(para), file=log_file)
    if freezed_too is True:
        print("\n Following parameters of the model are freezed:", file=log_file)
        for para in freezed:
            print("- {}".format(para), file=log_file)
