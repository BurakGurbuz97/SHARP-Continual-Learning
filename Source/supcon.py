import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional




class SupConLoss(nn.Module):
    def __init__(self, args: ...):
        super(SupConLoss, self).__init__()
        self.temperature = args.supcon_temperature
        self.args = args

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        # mask[i, j] == 1 if label_i == label_j
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = features
        anchor_feature = contrast_feature

        # Compute all pairwise inner products and normalize with temperature
        anchor_feature_norm = torch.nn.functional.normalize(anchor_feature, p=2, dim=1)
        contrast_feature_norm = torch.nn.functional.normalize(contrast_feature, p=2, dim=1)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature_norm, contrast_feature_norm.T),
            self.temperature)
        
        # for numerical stability substract max product (suggested in original supervised contrative loss paper)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases (diagonal values)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        # Since LTM does not have positive, use self as only pos
        # It does not matter since features are coming from LTM and does
        # not have backprob function
        mask = mask * logits_mask
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        # Substract (divide inside log) sum over A(i) from each inner product
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()


        return loss