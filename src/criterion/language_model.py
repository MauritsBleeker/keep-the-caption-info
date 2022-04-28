import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, targets, masks):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        targets = targets[:, 1:]
        masks = masks[:, 1:]
        logits = logits[:,:-1]

        # truncate to the same size
        batch_size = logits.shape[0]
        targets = targets[:, :logits.shape[1]]
        masks = masks[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        targets = targets.contiguous().view(-1)
        masks = masks.contiguous().view(-1)
        loss = self.loss_fn(logits, targets)
        output = torch.sum(loss * masks) / batch_size

        return output
