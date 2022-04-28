import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class InputDecoder(nn.Module):

	def __init__(self, output_size, config):
		"""
		Reconstruct the original input caption
		:param output_size:
		:param config:
		"""
		super(InputDecoder, self).__init__()

		self.config = config

		self.embed = nn.Embedding(output_size, self.config.model.caption_encoder.word_dim)
		self.gru = nn.GRU(self.config.model.caption_encoder.word_dim, self.config.model.embed_dim)
		self.token_prediction_layer = nn.Linear(self.config.model.embed_dim, output_size)

	def forward(self, z_captions, targets, lengths):
		"""

		:param z_captions:
		:param targets:
		:param lengths:
		:return:
		"""

		wemb_out = self.embed(targets)

		packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)

		hidden, _ = self.gru(packed, z_captions.unsqueeze(0))

		hidden = pad_packed_sequence(hidden, batch_first=True)[0]

		logist = self.token_prediction_layer(hidden)
		seq_logprob = F.log_softmax(logist, dim=1)

		return seq_logprob
