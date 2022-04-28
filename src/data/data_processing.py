import os
from collections import defaultdict
import pickle
import time


def process_dataset(json_file, img_folder, dset_name, root, k=5):
	"""

	:param json_file:
	:param img_folder:
	:param dset_name:
	:param root:
	:param k:
	:return:
	"""
	train_set = defaultdict(dict)
	validation_set = defaultdict(dict)
	test_set = defaultdict(dict)

	t = time.time()

	try:
		for i, image in enumerate(json_file['images']):

			if image['split'] == 'train' or image['split'] == 'restval':
				split = train_set
			elif image['split'] == 'val':
				split = validation_set
			elif image['split'] == 'test':
				split = test_set
			else:
				raise Exception('Unkown split')

			if dset_name == 'coco':
				folder = '/' + image['filepath']
			else:
				folder = ''

			img = open('{}/{}{}/{}'.format(root, img_folder, folder, image['filename']), 'rb').read()

			image_object = {
				'image': img,
				'filename': image['filename'],
				'caption_ids': image['sentids'],
				'imgid': image['imgid']
			}

			if dset_name == 'coco':
				image_object['cocoid'] = image['cocoid']

			split['images'][image['imgid']] = image_object

			if len(image['sentences']) != k:
				print("Tuple with {} captions, {}".format(len(image['sentences']), image['imgid']))

			for caption in image['sentences'][:k]:

				caption_object = {
					'caption': caption['raw'],
					'imgid': caption['imgid'],
					'tokens': caption['tokens'],
					'sentid': caption['sentid'],
					'filename': image['filename'],
				}

				if dset_name == 'coco':
					caption_object['cocoid'] = image['cocoid']

				split['captions'][caption['sentid']] = caption_object

			if i % 1000 == 0 and i > 0:
				print("Number of images processed: {}, in {} sec.".format(i, time.time() - t))
				t = time.time()

	except Exception:

		raise Exception('Error during processing')

	print("Size train set:{}".format(len(train_set['images'])))
	print("Size validation set:{}".format(len(validation_set['images'])))
	print("Size test set:{}".format(len(test_set['images'])))

	pickle.dump(train_set, open(os.path.join(root, 'training_set.pickle'), "wb"))
	pickle.dump(validation_set, open(os.path.join(root, 'validation_set.pickle'), "wb"))
	pickle.dump(test_set, open(os.path.join(root, 'test_set.pickle'), "wb"))


if __name__ == "__main__":

	json_file = 'annotations/f30k/dataset_flickr30k.json'
	img_folder = '<folder were the images are stored>'
	dset_name = 'f30k'
	root = '<root folder of the images>'

	process_dataset(json_file, img_folder, dset_name, root, k=5)
