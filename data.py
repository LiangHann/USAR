from os.path import join

from dataset import MICROSCOPY, NATURAL


def get_training_set(dataset, root_dir):
    if dataset=='MICROSCOPY':
        return MICROSCOPY(root_dir, True)
    elif dataset=='NATURAL':
        return NATURAL(root_dir, True)

def get_test_set(dataset, root_dir):
    if dataset=='MICROSCOPY':
        return MICROSCOPY(root_dir, False)
    elif dataset=='NATURAL':
        return NATURAL(root_dir, False)
