import torch.nn as nn
from utils.utils import cosine_sim


class InfoNCE(nn.Module):

    def __init__(self, tau=0.1):
        """

        :param tau:
        """
        super(InfoNCE, self).__init__()

        self.loss = nn.LogSoftmax(dim=1)
        self.tau = tau

    def forward(self, images, captions):
        """

        :param images:
        :param captions:
        :return:
        """
        t2i = cosine_sim(captions, images) / self.tau
        i2t = cosine_sim(images, captions) / self.tau

        image_retrieval_loss = - self.loss(t2i).diag().mean()
        caption_retrieval_loss = - self.loss(i2t).diag().mean()

        loss = 0.5 * caption_retrieval_loss + 0.5 * image_retrieval_loss

        return loss

