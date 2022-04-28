import torch
import torch.nn as nn
from models.image_encoder import ImageEncoder
import logging
from models.projection_head import ProjectionHead


class ImageCaptionEncoder(nn.Module):

	def __init__(self, config, init_weights=True):
		"""

		:param config:
		:param vocab:
		"""
		super(ImageCaptionEncoder, self).__init__()

		self.config = config

		self.image_encoder = ImageEncoder(config=self.config, init_weights=init_weights)

		if self.config.model.caption_encoder.tune_targets:
			self.caption_encoder = ProjectionHead(in_features=self.config.model.embed_dim, projection_dim=self.config.model.embed_dim)

		self.iteration = 0

		self.img_encoder_device = self.cap_encoder_device = 'cpu'

		self.to_devide()

	def forward(self, images, caption_target):
		"""

		:param images:
		:param caption_target:
		:return:
		"""
		self.iteration += 1

		z_images = self.image_encoder(images)

		if self.config.model.caption_encoder.tune_targets:
			caption_target = self.caption_encoder(caption_target)

		return z_images, caption_target

	def finetune(self, image_encoder=True):
		"""

		:param image_encoder:
		:return:
		"""

		logging.info("Start fine tuning encoders")

		if image_encoder:
			self.image_encoder.finetune()

	def to_devide(self):
		"""

		:return:
		"""
		if torch.cuda.is_available():

			self.to('cuda')

			self.img_encoder_device = 'cuda'
			self.cap_encoder_device = 'cuda'

		else:

			logging.info("Using CPU")
			self.to('cpu')
