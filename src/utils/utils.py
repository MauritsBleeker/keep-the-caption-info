import json
import os
import torch.nn.functional as F
from operator import attrgetter


def load_json(root, file):
    """

    :param root:
    :param file:
    :return:
    """
    return json.load(open(os.path.join(root, file)))


def l2_normalize(tensor, axis=-1):
    """
    L2-normalize columns of tensor
    :param tensor:
    :param axis:
    :return:
    """
    return F.normalize(tensor, p=2, dim=axis)


def cosine_sim(x, y):
    """
    Cosine similarity
    :param x:
    :param y:
    :return:
    """
    return x.mm(y.t())


def update_config(config, kwargs):
    """
    Update config with flags from the commandline
    :param config:
    :param kwargs:
    :return:
    """

    for key, value in kwargs.items():
        try:
            _ = attrgetter(key)(config)
            subconfig = config

            for sub_key in key.split('.'):
                if isinstance(subconfig[sub_key], dict):
                    subconfig = subconfig[sub_key]
                else:
                    if isinstance(subconfig[sub_key], type(value)):
                        subconfig[sub_key] = value
                    else:
                        raise Exception("wrong value type")

        except AttributeError:
            print("{} not in config".format(key))

    return config
