import fire
import munch
import torch
from munch import Munch
from data.dataset import Dataset
from data.dataset import collate_fn
from torch.utils.data import DataLoader
from models.model import ImageCaptionEncoder
from criterion.target_reconstruction import TargetReconstruction
from criterion.info_nce import InfoNCE
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

		self.config = config

		logging.info("Loading {} from: {}".format(
			self.config.dataset.train_pickle,
			self.config.dataset.root)
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

		self.model = ImageCaptionEncoder(config=self.config)

		logging_gradients = [self.model.image_encoder]

		if self.config.model.caption_encoder.tune_targets:
			logging_gradients.append(self.model.caption_encoder)

		wandb.watch(
			logging_gradients,
			log='all',
			log_freq=self.config.train.log_step,
			idx=0
		)

		if self.config.model.caption_encoder.tune_targets:
			self.criterion = InfoNCE(tau=self.config.criterion.tau)
		else:
			self.criterion = TargetReconstruction()

		params = [{'params': self.model.image_encoder.parameters(), 'lr': self.config.optimizer.learning_rate}]

		if self.config.model.caption_encoder.tune_targets:
			params.append({'params': self.model.caption_encoder.parameters(), 'lr': 2e-5})

		self.optimizer = get_optimizer(
			optimizer_name=self.config.optimizer.name,
			parameters=params,
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
				)

		self.store_model(file_name=self.config.train.model_save_path)

	def training_epoch(self):
		"""

		:return:
		"""

		self.start_training()

		if self.config.optimizer.weight_averaging.use_weight_averaging:
			self.weight_average = SWA(
				self.optimizer,
				swa_start=int(self.config.optimizer.weight_averaging.percentage * len(self.dataloader)),
				swa_freq=int(((1 - self.config.optimizer.weight_averaging.percentage) * len(self.dataloader))/self.config.optimizer.weight_averaging.checkpoints)
			)

		for i, (images, targets, caption_ids, image_ids, idx) in enumerate(self.dataloader):

			loss = self.iteration(images, targets)

			if i % self.config.train.log_step == 0:

				self.logging(i=i, loss=loss)

			self.step += 1

		if self.config.optimizer.weight_averaging.use_weight_averaging:

			logging.info('--- Apply SWA ---')
			self.weight_average.swap_swa_sgd()

			logging.info('Epoch: [{0}][{1}/{2}]\t''Loss value: {3}\t'.format(self.epoch, i, len(self.dataloader), loss.data))
			wandb.log({
				'epoch': self.epoch,
				'step': self.step,
				'loss': loss.data,
				'lr': self.optimizer.param_groups[0]['lr']
				}, step=self.step)

			self.step += 1

	def iteration(self, images, caption_target):
		"""

		:param images:
		:param caption_target:
		:return:
		"""

		self.optimizer.zero_grad()

		if self.config.train.use_fp16:
			with amp.autocast():

				z_images, caption_target = self.model(
					images.to(self.model.img_encoder_device),
					caption_target.to(self.model.cap_encoder_device)
				)

				loss = self.compute_loss(z_images, caption_target)

			self.scaler.scale(loss).backward()
		else:
			z_images, caption_target = self.model(
				images.to(self.model.img_encoder_device),
				caption_target.to(self.model.cap_encoder_device)
			)

			loss = self.compute_loss(z_images, caption_target)

			loss.backward()

		self.optimizer_step()

		return loss

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

			self.scaler.update()
		else:
			if self.config.optimizer.weight_averaging.use_weight_averaging:
				self.weight_average.step()
			else:
				self.optimizer.step()

	def compute_loss(self, z_images, targets):
		"""

		:param z_images:
		:param targets:
		:param targets:
		:return:
		"""

		loss = self.criterion(z_images, targets)

		return loss

	def logging(self, i, loss):
		"""

		:param i
		:param loss: total loss value
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

		return loss

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

		self.optimizer.param_groups[0]['lr'] = self.lr

	def store_model(self, file_name):

		state_dict = {
			'model': self.model.state_dict(),
			'criterion': self.criterion.state_dict(),
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
		tags=[config.dataset.dataset_name, 'paper_experiments', 'direct_target']
	)

	trainer = Trainer(config=config)

	trainer.train()


if __name__ == '__main__':
	fire.Fire(main)
