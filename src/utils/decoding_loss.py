import torch
import torch.nn as nn
from criterion.target_reconstruction import TargetReconstruction
from criterion.language_model import LanguageModelCriterion
from utils.constraint import Constraint
from utils.constraint import ConstraintOptimizer


class DecodingLoss(nn.Module):
	"""
	Class that computes the decoding loss
	"""
	def __init__(self, config):
		"""

		:param self:
		:param config:
		:return:
		"""
		super(DecodingLoss, self).__init__()

		self.config = config

		if self.config.model.target_decoder.input_decoding:
			self.reconstruction_criterion = LanguageModelCriterion()
		else:
			self.reconstruction_criterion = TargetReconstruction(
				reconstruction_metric=self.config.criterion.reconstruction_metric
			)

		if self.config.recconstruction_constraint.use_constraint:
			self.reconstruction_constraint = Constraint(
				self.config.recconstruction_constraint.bound,
				'le',
				start_val=float(self.config.recconstruction_constraint.start_val),
			)

			self.constraint_opt = ConstraintOptimizer(
				torch.optim.SGD,
				params=self.reconstruction_constraint.parameters(),
				lr=5e-3,
				momentum=self.config.recconstruction_constraint.alpha,
				dampening=self.config.recconstruction_constraint.alpha,
			)

	def forward(self, reconstructions, targets, mask):
		"""

		:param reconstructions: predicted reconstruction
		:param targets: targets, either latent or input
		:param mask: tokens mask, only for input decoding
		:return:
		"""

		if self.config.model.target_decoder.input_decoding:
			reconstruction_loss = self.reconstruction_criterion(reconstructions, targets.long(), mask)
		else:
			reconstruction_loss = self.reconstruction_criterion(reconstructions, targets)

		if self.config.recconstruction_constraint.use_constraint:
			constraint_loss = self.reconstruction_constraint(reconstruction_loss)
			return constraint_loss, reconstruction_loss
		else:
			return reconstruction_loss, reconstruction_loss

