import torch
import torch.nn as nn
from models.image_encoder import ImageEncoder
from models.caption_encoder import CaptionEncoder
from models.target_decoder import TargetDecoder
from models.input_decoder import InputDecoder
import logging


class ImageCaptionEncoder(nn.Module):

	def __init__(self, config, vocab, init_weights=True):
		"""

		:param config:
		:param vocab:
		"""
		super(ImageCaptionEncoder, self).__init__()

		self.config = config

		self.image_encoder = ImageEncoder(config=self.config, init_weights=init_weights)

		self.caption_encoder = CaptionEncoder(word2idx=vocab.word2idx, config=self.config, init_weights=init_weights)

		if self.config.model.target_decoder.decode_target:
			if self.config.model.target_decoder.input_decoding:
				self.input_decoder = InputDecoder(
					output_size=len(vocab.word2idx),
					config=self.config
				)
			else:
				self.target_decoder = TargetDecoder(
					in_features=self.config.model.embed_dim,
					hidden_features=self.config.model.target_decoder.hidden_features,
					reconstruction_dim=self.config.model.target_decoder.reconstruction_dim
				)

		self.iteration = 0

		self.img_encoder_device = self.cap_encoder_device = 'cpu'

		self.to_devide()

	def forward(self, images, caption, cap_lengths):
		"""

		:param images:
		:param caption:
		:param cap_lengths:
		:return:
		"""
		self.iteration += 1

		z_images = self.image_encoder(images)
		z_captions = self.caption_encoder(caption, cap_lengths, device=self.cap_encoder_device)

		reconstructions = None

		if self.config.model.target_decoder.decode_target:
			if self.config.model.target_decoder.input_decoding:
				reconstructions = self.input_decoder(z_captions=z_captions, targets=caption, lengths=cap_lengths)
			else:
				reconstructions = self.target_decoder(z_captions)

		return z_images, z_captions, reconstructions

	def finetune(self, image_encoder=True, caption_encoder=True):
		"""

		:param image_encoder:
		:param caption_encoder:
		:return:
		"""

		logging.info("Start fine tuning encoders")

		if caption_encoder:
			self.caption_encoder.finetune()
		if image_encoder:
			self.image_encoder.finetune()

	def to_devide(self):
		"""

		:return:
		"""
		if torch.cuda.is_available():
			if torch.cuda.device_count() == 2:
				logging.info("Using two GPUs")

				self.img_encoder_device = 'cuda:0'
				self.cap_encoder_device = 'cuda:1'

				self.caption_encoder.to(self.cap_encoder_device)
				self.image_encoder.to(self.img_encoder_device)

				if self.config.model.target_decoder.decode_target:
					if self.config.model.target_decoder.input_decoding:
						self.input_decoder.to(self.cap_encoder_device)
					else:
						self.target_decoder.to(self.cap_encoder_device)

			else:
				logging.info("Using one GPU")
				self.to('cuda')

				self.img_encoder_device = self.cap_encoder_device = 'cuda'

		else:
			logging.info("Using CPU")
			self.to('cpu')
