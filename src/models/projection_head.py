import torch.nn as nn


class ProjectionHead(nn.Module):

	def __init__(self, in_features, projection_dim):
		"""

		:param in_features:
		:param projection_dim:
		"""
		super(ProjectionHead, self).__init__()

		self.projector = nn.Sequential(
			nn.Linear(in_features, in_features, bias=False),
			nn.ReLU(),
			nn.Linear(in_features, projection_dim, bias=False),
		)

	def forward(self, x):
		"""

		:param x:
		:return:
		"""
		return self.projector(x)

	def init_weights(self):
		"""

		:return:
		"""

		nn.init.xavier_uniform_(self.projector[0].weight)
		nn.init.xavier_uniform_(self.projector[2].weight)
