import warnings
import torch.optim as optim
from adamp import AdamP


def get_optimizer(optimizer_name, parameters, config, logger=None):
    """

    :param optimizer_name:
    :param parameters:
    :param config:
    :param logger:
    :return:
    """
    if logger:
        logger.log('creating [{}] from Config({})'.format(optimizer_name, config))
    if optimizer_name == 'adam':
        if set(config.keys()) - {'learning_rate', 'betas', 'eps',
                                 'weight_decay', 'amsgrad', 'name'}:
            warnings.warn('found unused keys in {}'.format(config.keys()))
        optimizer = optim.Adam(parameters,
                               lr=config.learning_rate,
                               betas=config.get('betas', (0.9, 0.999)),
                               eps=float(config.get('eps', 1e-8)),
                               weight_decay=float(config.get('weight_decay', 0)),
                               amsgrad=config.get('amsgrad', False))
    elif optimizer_name == 'adamn' or optimizer_name == 'adamp':
        if set(config.keys()) - {'learning_rate', 'betas', 'eps',
                                 'weight_decay', 'name'}:
            warnings.warn('found unused keys in {}'.format(config.keys()))
        optimizer = AdamP(parameters,
                          #lr=config.learning_rate,
                          betas=config.get('betas', (0.9, 0.999)),
                          eps=float(config.get('eps', 1e-8)),
                          weight_decay=float(config.get('weight_decay', 0)))
    else:
        raise ValueError(f'Invalid optimizer name: {optimizer_name}')
    return optimizer


def get_lr_scheduler(scheduler_name, optimizer, config, logger=None):
	"""

	:param scheduler_name:
	:param optimizer:
	:param config:
	:param logger:
	:return:
	"""
	if logger:
		logger.log('creating [{}] from Config({})'.format(scheduler_name, config))

	if scheduler_name == 'cosine_annealing':
		lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.lr_scheduler.T_max)
	elif scheduler_name == 'multi_step_lr':
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_scheduler.milestones, gamma=config.train.finetune_lr_decay)
	else:
		raise ValueError(f'Invalid scheduler name: {scheduler_name}')
	return lr_scheduler
