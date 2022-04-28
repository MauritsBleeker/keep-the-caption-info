import fire
import munch
import torch
from munch import Munch
from data.dataset import Dataset
from data.dataset import collate_fn
from torch.utils.data import DataLoader
from models.model import ImageCaptionEncoder
from criterion.info_nce import InfoNCE
from criterion.triplet import Triplet
from criterion.barlow_twins import BarlowTwinsLoss
from utils.decoding_loss import DecodingLoss
import torch.nn as nn
from evaluate import Evaluator
try:
	from torch.cuda import amp
except ImportError:
	print('failed to import amp')
import wandb
from utils.optimizers import get_optimizer
from utils.optimizers import get_lr_scheduler
import os
import logging
import torch.backends.cudnn as cudnn
from utils.utils import update_config
from torchcontrib.optim import SWA


class Trainer(object):

	def __init__(self, config):
		"""

		:param config: Config class
		"""

		self.config = config

		logging.info("Loading {} from: {}".format(
			self.config.dataset.train_pickle,
			self.config.dataset.root
			)
		)

		self.trainset = Dataset(
			pickle_file=self.config.dataset.train_pickle,
			split='train',
			config=self.config
		)

		self.dataloader = DataLoader(
			self.trainset,
			shuffle=True,
			batch_size=self.config.dataloader.batch_size,
			num_workers=self.config.dataloader.num_workers,
			collate_fn=collate_fn,
			pin_memory=True
		)

		self.model = ImageCaptionEncoder(vocab=self.trainset.vocab, config=self.config)

		logging_gradients = [self.model.caption_encoder, self.model.image_encoder]

		if self.config.model.target_decoder.decode_target:
			if self.config.model.target_decoder.input_decoding:
				logging_gradients.append(self.model.input_decoder)
			else:
				logging_gradients.append(self.model.target_decoder)

		wandb.watch(
			logging_gradients,
			log='all',
			log_freq=self.config.train.log_step,
			idx=0
		)

		assert self.config.criterion.name == 'infonce' or self.config.criterion.name == 'triplet' or self.config.criterion.name == 'barlow'

		if self.config.criterion.name == 'infonce':
			self.contrastive_criterion = InfoNCE(self.config.criterion.tau)
		elif self.config.criterion.name == 'triplet':
			self.contrastive_criterion = Triplet(margin=self.config.criterion.margin, max_violation=True)
		elif self.config.criterion.name == 'barlow':
			# not used for paper experiments
			self.contrastive_criterion = BarlowTwinsLoss(embed_dim=self.config.model.embed_dim, device=self.model.cap_encoder_device)
		else:
			raise NotImplementedError("Loss is not implemented")

		if self.config.model.target_decoder.decode_target:

			self.decoding_loss = DecodingLoss(config=self.config)

		self.optimizer = get_optimizer(
			optimizer_name=self.config.optimizer.name,
			parameters=self.model.parameters(),
			config=self.config.optimizer
		)

		self.weight_average = None

		self.lr = self.config.optimizer.learning_rate

		self.lr_scheduler = get_lr_scheduler(
			scheduler_name=self.config.lr_scheduler.name,
			optimizer=self.optimizer,
			config=self.config
		)

		if self.config.train.use_fp16:

			logging.info("Using FP16 for training")
			self.scaler = amp.GradScaler()

		self.evaluator = Evaluator(model=self.model, split='val', config=self.config)

		self.best_score = 0
		self.epoch = 0
		self.step = 0

	def train(self):
		"""
		run training

		:return:
		"""
		logging.info('--- Start training ---')

		for epoch in range(self.config.train.n_epochs):

			self.training_epoch()

			if epoch % self.config.train.val_epochs == 0:

				logging.info('--- Start evaluation ---')

				rsum = self.evaluator.evaluate(step=self.step)

				if rsum > self.best_score:

					logging.info("Store model in epoch {}".format(epoch))
					self.best_score = rsum
					self.store_model(file_name=self.config.train.best_model_save_path)

				wandb.log({
					'best_score': self.best_score,
				}, step=self.step)

			self.lr_scheduler.step()

			self.epoch = epoch + 1

			if epoch == self.config.lr_scheduler.T_max - 1:

				logging.info('--- Start fine-tuning ---')

				if self.config.lr_scheduler.name == 'cosine_annealing':

					self.lower_init_lr()

					self.lr_scheduler = get_lr_scheduler(
						scheduler_name=self.config.lr_scheduler.name,
						optimizer=self.optimizer,
						config=self.config
					)

				self.model.finetune(
					image_encoder=self.config.model.image_encoder.img_finetune,
					caption_encoder=self.config.model.caption_encoder.txt_finetune,
				)

		self.store_model(file_name=self.config.train.model_save_path)

	def training_epoch(self):
		"""
		run one training epoch

		:return:
		"""

		self.start_training()

		if self.config.optimizer.weight_averaging.use_weight_averaging:

			self.weight_average = SWA(
				self.optimizer,
				swa_start=int(self.config.optimizer.weight_averaging.percentage * len(self.dataloader)),
				swa_freq=int(((1 - self.config.optimizer.weight_averaging.percentage) * len(self.dataloader))/self.config.optimizer.weight_averaging.checkpoints)
			)

		for i, (images, tokens, cap_lengths, targets, caption_ids, image_ids, idx, mask) in enumerate(self.dataloader):

			loss, contrastive_loss, reconstruction_loss = self.iteration(images, tokens, targets, cap_lengths, mask)

			if i % self.config.train.log_step == 0:

				self.logging(i=i, loss=loss, contrastive_loss=contrastive_loss, reconstruction_loss=reconstruction_loss)

			self.step += 1

		if self.config.optimizer.weight_averaging.use_weight_averaging:

			logging.info('--- Apply SWA ---')
			self.weight_average.swap_swa_sgd()

	def iteration(self, images, tokens, targets, cap_lengths, mask=None):
		"""
		training iteration

		:param images: input images
		:param tokens: input captions
		:param targets: caption targets
		:param cap_lengths: length of each caption
		:param mask: mask out padding tokens
		:return: loss, contrastive_loss, reconstruction_loss
		"""
		self.optimizer.zero_grad()

		if self.config.recconstruction_constraint.use_constraint:
			self.decoding_loss.constraint_opt.zero_grad()

		if self.config.model.target_decoder.input_decoding:
			targets = tokens

		if self.config.train.use_fp16:

			with amp.autocast():

				z_images, z_captions, reconstructions = self.model(
					images.to(self.model.img_encoder_device),
					tokens.to(self.model.cap_encoder_device),
					cap_lengths,
				)

				loss, contrastive_loss, reconstruction_loss = self.compute_loss(z_images, z_captions, reconstructions, targets, mask)

			self.scaler.scale(loss).backward()

		else:

			z_images, z_captions, reconstructions = self.model(
				images.to(self.model.img_encoder_device),
				tokens.to(self.model.cap_encoder_device),
				cap_lengths
			)

			loss, contrastive_loss, reconstruction_loss = self.compute_loss(z_images, z_captions, reconstructions, targets, mask)

			loss.backward()

		self.optimizer_step()

		return loss, contrastive_loss, reconstruction_loss

	def optimizer_step(self):
		"""

		:return:
		"""

		if self.config.train.grad_clip > 0:

			if self.config.train.use_fp16:

				self.scaler.unscale_(self.optimizer)

			nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)

		if self.config.train.use_fp16:
			if self.config.optimizer.weight_averaging.use_weight_averaging:
				self.scaler.step(self.weight_average)
			else:
				self.scaler.step(self.optimizer)

			if self.config.model.target_decoder.decode_target and self.config.recconstruction_constraint.use_constraint:
				self.scaler.step(self.decoding_loss.constraint_opt)

			self.scaler.update()
		else:
			if self.config.optimizer.weight_averaging.use_weight_averaging:
				self.weight_average.step()
			else:
				self.optimizer.step()

			if self.config.model.target_decoder.decode_target and self.config.recconstruction_constraint.use_constraint:
				self.decoding_loss.constraint_opt.step()

	def compute_loss(self, z_images, z_captions, reconstructions=None, targets=None, mask=None):
		"""
		compute loss value

		:param z_images: image representations
		:param z_captions: caption representations
		:param reconstructions: reconstruction of the caption
		:param targets:
		:return: loss, contrastive_loss, reconstruction_loss_value
		"""

		if torch.cuda.device_count() == 2:

			z_images = z_images.to(self.model.cap_encoder_device)
			z_captions = z_captions.to(self.model.cap_encoder_device)

			if reconstructions is not None:
				targets = targets.to(self.model.cap_encoder_device)
				reconstructions = reconstructions.to(self.model.cap_encoder_device)

				if self.config.model.target_decoder.input_decoding:
					mask = mask.to(self.model.cap_encoder_device)

		elif torch.cuda.device_count() == 1:
			targets = targets.to('cuda')

			if self.config.model.target_decoder.input_decoding:
				mask = mask.to('cuda')

		contrastive_loss = self.contrastive_criterion(z_images, z_captions)

		reconstruction_loss_value = None

		if reconstructions is not None and targets is not None:

			reconstruction_loss, reconstruction_loss_value = self.decoding_loss(reconstructions, targets, mask)
			loss = reconstruction_loss + contrastive_loss

		else:
			loss = contrastive_loss

		return loss, contrastive_loss, reconstruction_loss_value

	def logging(self, i, loss, contrastive_loss, reconstruction_loss):
		"""

		:param loss: total loss value
		:param contrastive_loss:
		:param reconstruction_loss:
		:return:
		"""

		logging.info(
			'Epoch: [{0}][{1}/{2}]\t''Loss value: {3}\t'.format(self.epoch, i, len(self.dataloader), loss.data)
		)

		wandb.log({
			'epoch': self.epoch,
			'step': self.step,
			'loss': loss.data,
			'lr': self.optimizer.param_groups[0]['lr']
		}, step=self.step)

		if reconstruction_loss:

			wandb.log({
				'contrastive_loss': contrastive_loss.data.data,
				'reconstruction_loss': reconstruction_loss.data,
			}, step=self.step)

			if self.config.recconstruction_constraint.use_constraint:

				wandb.log({
					'multiplier': self.decoding_loss.reconstruction_constraint.multiplier.data
				}, step=self.step)

	def start_training(self):
		"""

		:return:
		"""

		self.model.train()

	def lower_init_lr(self, decay=0.1):
		"""

		:param decay: lr decay
		:return:
		"""
		self.lr *= decay

		for g in self.optimizer.param_groups:
			g['lr'] = self.lr

	def store_model(self, file_name):
		"""

		:param file_name: file name of the stored model
		:return:
		"""

		state_dict = {
			'model': self.model.state_dict(),
			'criterion': self.contrastive_criterion.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'lr_scheduler': self.lr_scheduler.state_dict(),
			'config': munch.unmunchify(self.config),
			'word2idx': self.trainset.vocab.word2idx,
			'step': self.step,
			'vocab': self.trainset.vocab
		}

		directory = os.path.join(self.config.experiment.out_dir, self.config.experiment.experiment_name)

		if not os.path.exists(directory):
			os.makedirs(directory)

		torch.save(state_dict, os.path.join(directory, file_name))


def main(yaml_file, **kwargs):
	"""

	:param yaml_file: config yaml file
	:param kwargs:
	:return:
	"""

	config = Munch.fromYAML(open(yaml_file, 'rb'))

	if kwargs:
		config = update_config(config, kwargs)

	logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

	cudnn.benchmark = True

	wandb.init(
		project=config.experiment.wandb_project,
		entity='<WandB user name>',
		name=config.experiment.experiment_name,
		dir=config.experiment.wandb_dir,
		config=munch.unmunchify(config),
		tags=[config.dataset.dataset_name, 'paper_experiments']
	)

	trainer = Trainer(config=config)

	trainer.train()


if __name__ == '__main__':
	fire.Fire(main)
