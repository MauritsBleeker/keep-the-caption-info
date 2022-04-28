"""
Image encoder based on PCME implementation.
Reference code:
https://github.com/naver-ai/pcme/blob/main/models/caption_encoder.py
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchtext
from utils.utils import l2_normalize
from models.projection_head import ProjectionHead
import logging


class CaptionEncoder(nn.Module):

	def __init__(self, word2idx, config, init_weights=True):
		"""

		:param word2idx:
		:param config:
		"""
		super(CaptionEncoder, self).__init__()

		self.config = config

		self.wemb_type = self.config.model.caption_encoder.wemb_type
		self.word_dim = self.config.model.caption_encoder.word_dim
		self.embed_dim = self.config.model.embed_dim

		self.embed = nn.Embedding(len(word2idx), self.word_dim)
		self.embed.weight.requires_grad = self.config.model.caption_encoder.tune_from_start

		# Sentence embedding
		self.rnn = nn.GRU(self.word_dim, self.embed_dim // 2, bidirectional=True, batch_first=True)

		self.fc = ProjectionHead(in_features=self.embed_dim, projection_dim=self.embed_dim)

		if self.config.criterion.name == 'triplet':
			self.bn = nn.BatchNorm1d(self.embed_dim)

		if init_weights:
			self.init_weights(self.wemb_type, word2idx, self.word_dim, cache_dir=self.config.experiment.cache_dir)
			self.fc.init_weights()

	def init_weights(self, wemb_type, word2idx, word_dim, cache_dir):
		"""

		:param wemb_type:
		:param word2idx:
		:param word_dim:
		:param cache_dir:
		:return:
		"""
		if wemb_type is None:
			nn.init.xavier_uniform_(self.embed.weight)
		else:
			# Load pretrained word embedding
			if 'fasttext' == wemb_type.lower():
				wemb = torchtext.vocab.FastText(cache=cache_dir)
			elif 'glove' == wemb_type.lower():
				wemb = torchtext.vocab.GloVe(cache=cache_dir)
			else:
				raise Exception('Unknown word embedding type: {}'.format(wemb_type))
			assert wemb.vectors.shape[1] == word_dim

			# quick-and-dirty trick to improve word-hit rate
			missing_words = []
			for word, idx in word2idx.items():
				if word not in wemb.stoi:
					word = word.replace('-', '').replace('.', '').replace("'", '')
					if '/' in word:
						word = word.split('/')[0]
				if word in wemb.stoi:
					self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
				else:
					missing_words.append(word)
			logging.info('Words: {}/{} found in vocabulary; {} words missing'.format(
				len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

	def forward(self, captions, lengths, device):
		"""

		:param captions:
		:param lengths:
		:return:
		"""
		# Embed word ids to vectors
		wemb_out = self.embed(captions)

		# Forward propagate RNNs
		packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
		if torch.cuda.device_count() > 1:
			self.rnn.flatten_parameters()
		rnn_out, _ = self.rnn(packed)
		padded = pad_packed_sequence(rnn_out, batch_first=True)

		# Reshape *final* output to (batch_size, hidden_size)
		I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
		if torch.cuda.is_available():
			I = I.to(device)

		out = torch.gather(padded[0], 1, I).squeeze(1)

		out = self.fc(out)

		if self.config.criterion.name == 'triplet':
			out = self.bn(out)

		out = l2_normalize(out)

		return out

	def finetune(self):
		"""

		:return:
		"""
		self.embed.weight.requires_grad = True
