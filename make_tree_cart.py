from regression_tree_cart import *


def make_cart_tree(data, B, max_depth=500, Nmin=5, labels={}):
    n = len(data)
    root = grow_tree(data, 0, max_depth=max_depth, Nmin=Nmin, labels=labels,
                     start=True, feat_bag=True)
    return root
