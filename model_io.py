import pickle

from make_tree_cart import *


def save_model(filename, tree):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(tree, f)
    except Exception as e:
        print('save model error: ', e)


def load_model(filename):
    tree_root = None
    try:
        with open(filename, 'rb') as f:
            tree_root = pickle.load(f)
    except Exception as e:
        print('read model error: ', e)
    return tree_root


