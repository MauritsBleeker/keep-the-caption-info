"""
Reference code: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
"""
import torch
import torch.nn as nn


class BarlowTwinsLoss(torch.nn.Module):

	def __init__(self, embed_dim, device, lambda_param=0.0051):
		super(BarlowTwinsLoss, self).__init__()
		self.bn = nn.BatchNorm1d(embed_dim, affine=False)
		self.bn.to(device)
		self.lambd = lambda_param

	def forward(self, images, captions):
		# normalize repr. along the batch dimension

		c = self.bn(images).T @ self.bn(captions)

		# sum the cross-correlation matrix between all gpus
		c.div_(images.shape[0])

		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

		off_diag = self.off_diagonal(c).pow_(2).sum()

		loss = on_diag + self.lambd * off_diag

		return loss

	def off_diagonal(self, x):
		"""

		:param x:
		:return:
		"""

		# return a flattened view of the off-diagonal elements of a square matrix
		n, m = x.shape
		assert n == m
		return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
