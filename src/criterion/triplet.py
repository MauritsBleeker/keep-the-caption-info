"""
Reference code: https://github.com/fartashf/vsepp
"""

import torch.nn as nn
from utils.utils import cosine_sim
import torch
from torch.autograd import Variable


class Triplet(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2, max_violation=True):
        super(Triplet, self).__init__()
        self.margin = margin

        self.max_violation = max_violation

    def forward(self, images, caption):
        # compute image-sentence score matrix
        scores = cosine_sim(images, caption)
        diagonal = scores.diag().view(images.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.to(cost_s.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        loss = cost_s.sum() + cost_im.sum()

        return loss
