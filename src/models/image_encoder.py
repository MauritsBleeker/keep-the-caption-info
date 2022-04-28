"""
Image encoder based on PCME implementation.
Reference code:
https://github.com/naver-ai/pcme/blob/main/models/image_encoder.py
"""
import torch.nn as nn
from torchvision import models
from utils.utils import l2_normalize
from models.projection_head import ProjectionHead


class ImageEncoder(nn.Module):
    def __init__(self, config, init_weights=True):

        """

        :param config:
        """
        super(ImageEncoder, self).__init__()

        self.config = config
        embed_dim = self.config.model.embed_dim

        # Backbone CNN
        self.cnn = getattr(models, self.config.model.image_encoder.cnn_type)(pretrained=True)
        self.cnn_dim = self.cnn.fc.in_features

        self.avgpool = self.cnn.avgpool
        self.cnn.avgpool = nn.Sequential()

        self.fc = ProjectionHead(in_features=self.cnn_dim, projection_dim=embed_dim)

        self.cnn.fc = nn.Sequential()

        if self.config.criterion.name == 'triplet':
            self.bn = nn.BatchNorm1d(embed_dim)

        if init_weights:
            self.fc.init_weights()
            for idx, param in enumerate(self.cnn.parameters()):
                param.requires_grad = self.config.model.image_encoder.tune_from_start

    def forward(self, images):
        """

        :param images:
        :return:
        """
        out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
        pooled = self.avgpool(out_7x7).view(-1, self.cnn_dim)

        out = self.fc(pooled)

        if self.config.criterion.name == 'triplet':
            out = self.bn(out)

        out = l2_normalize(out)

        return out

    def finetune(self):
        """

        :return:
        """
        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = True
