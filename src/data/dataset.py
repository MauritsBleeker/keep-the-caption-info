from torch.utils.data import Dataset
import os
import pickle
from PIL import Image
from six import BytesIO as IO
from nltk.tokenize import word_tokenize
import random
import torch
from utils.transform import imagenet_transform
import logging


class Dataset(Dataset):

	def __init__(self, pickle_file, split, config):
		"""

		:param pickle_file:
		:param split:
		:param config:
		"""

		self.config = config
		self.split = split

		self.dataset = pickle.load(open(os.path.join(self.config.dataset.root, self.config.dataset.dataset_name, pickle_file),'rb'))

		self.image_transform = imagenet_transform(
			random_resize_crop=self.split == 'train',
			random_erasing_prob=self.config.dataloader.random_erasing_prob if self.split == 'train' else 0,
		)

		self.caption_ids = list(self.dataset['captions'])
		self.image_ids = list(self.dataset['images'])

		logging.info("Loading vocab: {}".format(self.config.dataset.vocab_file))
		self.vocab = pickle.load(open(os.path.join(self.config.dataset.root, self.config.dataset.dataset_name, 'vocab', self.config.dataset.vocab_file),'rb'))

		logging.info(f'Loaded {self.config.dataset.dataset_name} Split: {split},  n_images {len(self.image_ids)} n_captions {len(self.caption_ids)}')

	def __len__(self):

		return len(self.caption_ids)

	def __getitem__(self, idx):
		"""

		:param idx:
		:return:
		"""

		caption_id = self.caption_ids[idx]
		caption = self.dataset['captions'][caption_id]
		image_id = caption['imgid']

		image = Image.open(IO(self.dataset['images'][image_id]['image'])).convert('RGB')

		image = self.image_transform(image)

		tokens = self.tokenize(
			caption['caption'],
			self.vocab,
			caption_drop_prob=self.config.dataloader.caption_drop_prob if self.split == 'train' else 0
		)
		target = torch.Tensor(caption['target'])

		return image, tokens, caption_id, image_id, idx, target

	def tokenize(self, sentence, vocab, caption_drop_prob=0):
		"""
		nltk word_tokenize for caption transform.
		:param sentence:
		:param vocab:
		:param caption_drop_prob:
		:return:
		"""

		tokens = word_tokenize(str(sentence).lower())
		tokenized_sentence = list()
		tokenized_sentence.append(vocab('<start>'))

		if caption_drop_prob > 0:
			unk = vocab('<unk>')
			tokenized = [vocab(token) if random.random() > caption_drop_prob else unk for token in tokens]
		else:
			tokenized = [vocab(token) for token in tokens]

		if caption_drop_prob:

			N = int(len(tokenized) * caption_drop_prob)

			for _ in range(N):
				tokenized.pop(random.randrange(len(tokenized)))

		tokenized_sentence.extend(tokenized)
		tokenized_sentence.append(vocab('<end>'))

		return torch.Tensor(tokenized_sentence)


def collate_fn(data):
	"""

	:param data:
	:return:
	"""

	# Sort a data list by sentence length
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, captions, caption_ids, image_ids, idx, targets = zip(*data)

	# Merge images (convert tuple of 3D tensor to 4D tensor)
	images = torch.stack(images, 0)
	targets = torch.stack(targets, 0)
	# Merge sentences (convert tuple of 1D tensor to 2D tensor)
	cap_lengths = [len(cap) for cap in captions]
	tokens = torch.zeros(len(captions), max(cap_lengths)).long()

	mask = torch.zeros(len(captions), max(cap_lengths)).long()

	for i, cap in enumerate(captions):
		end = cap_lengths[i]
		tokens[i, :end] = cap[:end]
		mask[i, :end] = 1

	cap_lengths = torch.Tensor(cap_lengths).long()

	return images, tokens, cap_lengths, targets, caption_ids, image_ids, idx, mask
