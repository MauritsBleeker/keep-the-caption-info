import torch.nn as nn
import torch


class TargetReconstruction(nn.Module):

	def __init__(self, reconstruction_metric='cosine'):
		"""

		:param reconstruction_metric:
		"""

		super(TargetReconstruction, self).__init__()

		self.metric = reconstruction_metric
		self.cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

	def forward(self, reconstructions, targets):
		"""

		:param reconstructions:
		:param targets:
		:return:
		"""

		if self.metric == 'cosine':
			return (1 - self.cosine_sim(reconstructions, targets)).mean()
		elif self.metric == 'l2':
			return torch.cdist(reconstructions, targets, p=2).diag().mean()
		else:
			raise NotImplementedError
