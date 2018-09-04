import scipy
import scipy.optimize
import numpy
import copy
import sys
import pygame
import time
import random
import math
from bisect import bisect_right
from operator import itemgetter
r = 1
l = 2
BLACK = ( 0, 0,  0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
pygame.init()
tree_font = pygame.font.SysFont("Arial", 8)
height = 3000
width = 1200


def is_tree(node):
    return node is not None and node.is_tree()

def is_leaf(node):
    return node is None or not node.is_tree()

def feature_weights(path):
    weights = {}
    weights_percent = {}
    total = len(path) * 1.0
    for p in path:
        feature_idx, feature_val = p
        if feature_idx is None:
            continue
        weights[feature_idx] = weights.get(feature_idx, 0) + 1
        weights_percent[feature_idx] = weights[feature_idx] / total
    return weights, weights_percent

def data_split(data_dict, feature_idx, feature_split):
    data_left = {}
    data_right = {}
    for d in data_dict:
        if d[feature_idx] < feature_split:
            data_left[d] = data_dict[d]
        else:
            data_right[d] = data_dict[d]
    return data_left, data_right

def square_errors(data, anchor):
    data = numpy.array(list(data))
    return numpy.sum((data - anchor)**2)

class Tree(object):
    def __init__(self, error, predict, stdev, start, num_points):
        self.error = error
        self.predict = predict
        self.stdev = stdev
        self.start =  start
        self.split_var = None
        self.split_val = None
        self.split_lab = None
        self.left = None
        self.right = None
        self.num_points = num_points
        
    def lookup(self, x):
        """Returns the predicted value given the parameters."""
        if self.left == None:
            return self.predict
        if x[self.split_var] <= self.split_val:
            return self.left.lookup(x)
        return self.right.lookup(x)

    def lookup_with_path(self, x):
        """ Return the predicted val and split path given the parameters. """
        predict = None
        path = None
        if self.left is None:
            predict = self.predict
            path = []
        else:
            if x[self.split_var] <= self.split_val:
                predict, path = self.left.lookup_with_path(x)
                path.append( (self.split_var, self.split_val) )
            else:
                predict, path = self.right.lookup_with_path(x)
                path.append( (self.split_var, self.split_val) )
        return predict, path
    
    def predict_all(self, data):
        """Returns the predicted values for some list of data points."""
        return map(lambda x: self.lookup(x), data)
        
    
    def find_weakest(self):
        """Finds the smallest value of alpha and 
        the first branch for which the full tree 
        does not minimize the error-complexity measure."""
        if (self.right == None):
            return float("Inf"), [self]
        b_error, num_nodes = self.get_cost_params()
        alpha = (self.error - b_error) / (num_nodes - 1)
        alpha_right, tree_right = self.right.find_weakest()
        alpha_left, tree_left = self.left.find_weakest()
        smallest_alpha = min(alpha, alpha_right, alpha_left)
        smallest_trees = []
        # if there are multiple weakest links collapse all of them
        if smallest_alpha == alpha:
            smallest_trees.append(self)
        if smallest_alpha == alpha_right:
            smallest_trees = smallest_trees + tree_right
        if smallest_alpha == alpha_left:
            smallest_trees = smallest_trees + tree_left
        return smallest_alpha, smallest_trees
    
    def is_tree(self):
        return self.left is not None or self.right is not None

    def get_mean_predict(self):
        if is_tree(self.left):
            left_mean = self.left.get_mean_predict()
        else:
            left_mean = self.left.predict if self.left is not None else 0
        if is_tree(self.right):
            right_mean = self.right.get_,eam_predict()
        else:
            right_mean = self.right.predict if self.right is not None else 0
        return (right_mean + left_mean) / 2.0

    def get_feature_weights(self):
        """
        return the split counts for each feature
        """
        left_wts = None
        if self.left is not None:
            left_wts = self.left.get_feature_weights()
        else:
            left_wts = {}
        if self.right is not None:
            right_wts = self.right.get_feature_weights()
            for (k, v) in right_wts.items():
                left_wts[k] = left_wts.get(k, 0) + v
        if self.split_var is not None:
            left_wts[self.split_var] = left_wts.get(self.split_var, 0) + 1
        return left_wts

    def prune_cart_tree(self, test_data):
        """
        Prune cart tree, to merge some leavies
        test_data is dict([features]) -> value
        """
        if len(test_data) == 0:
            return self
        if is_tree(self.left) or is_tree(self.right):
            data_left, data_right = data_split(test_data, self.split_var, self.split_val)
            if is_tree(self.left):
                self.left = self.left.prune_cart_tree(data_left)
            if is_tree(self.right):
                self.right = self.right.prune_cart_tree(data_right)
        # left, right -> leaf
        if is_tree(self) and not is_tree(self.left) and not is_tree(self.right):
            data_left, data_right = data_split(test_data, self.split_var, self.split_val)
            left_predict = self.left.predict if self.left is not None else 0
            right_predict = self.right.predict if self.right is not None else 0
            error_no_merge = square_errors(data_left.values(), left_predict) + square_errors(data_right.values(), right_predict)
            mean_predicted = self.get_mean_predict()
            error_after_merge = square_errors(test_data.values(), mean_predicted)
            # merge left and right leaf, using mean predict
            if error_no_merge > error_after_merge:
                self.left = None
                self.right = None
                self.split_predict = mean_predicted
        return self
    
    def prune_tree(self):
        """Finds {a1, ..., ak} and {T1, ..., Tk},
        the sequence of nested subtrees from which to 
        choose the right sized tree."""
        trees = [copy.deepcopy(self)]
        alphas = [0]
        new_tree = copy.deepcopy(self)
        while 1:
            alpha, nodes = new_tree.find_weakest()
            for node in nodes:
                node.right = None
                node.left = None
            trees.append(copy.deepcopy(new_tree))
            alphas.append(alpha)
            # root node reached
            if (node.start == True):
                break
        return alphas, trees

    
    def get_cost_params(self):
        """Returns the branch error and number of nodes."""
        if self.right == None:
            return self.error, 1
        error, num_nodes = self.right.get_cost_params()
        left_error, left_num = self.left.get_cost_params()
        error += left_error
        num_nodes += left_num
        return error, num_nodes


    def get_length(self):
        """Returns the length of the tree."""
        if self.right == None:
            return 1
        right_len = self.right.get_length()
        left_len = self.left.get_length()
        return max(right_len, left_len) + 1

    def __drawTree(self, s_width, s_height, window, tree_length, horz_step, depth, d_nodes):
        """Draws an image of the tree using pygame."""
        branch = (s_width + horz_step, s_height)
        pygame.draw.line(window, BLACK, (s_width, s_height), branch)
        if self.right == None:
            p = tree_font.render("%.1f" % self.predict, 1, BLACK)
            window.blit(p, branch)
            return
        if self.split_lab != None:
            lab = tree_font.render("%s" % self.split_lab, 1, RED)    
        else:
            lab = tree_font.render("%d" % self.split_var, 1, RED)
        window.blit(lab, (s_width + (horz_step / 2), s_height))
        vert_step = d_nodes * 2 ** (tree_length - depth - 2)
        right_start = (s_width + horz_step, s_height - vert_step)
        left_start = (s_width + horz_step, s_height + vert_step)
        r_val = tree_font.render(">%d" % self.split_val, 1, RED)
        window.blit(r_val, right_start)
        l_val = tree_font.render("<%d" % self.split_val, 1, RED)
        window.blit(l_val, left_start)
        pygame.draw.line(window, BLACK, branch, right_start)
        pygame.draw.line(window, BLACK, branch, left_start)
        self.right.__drawTree(right_start[0], 
            right_start[1], window, tree_length, horz_step, depth + 1, d_nodes)
        self.left.__drawTree(left_start[0], 
            left_start[1], window, tree_length, horz_step, depth + 1, d_nodes)
        return

    
    def display_tree(self, save = False, filename = "image.jpg", view=True):
        """Wrapper function to draw the tree.
            If save is set to True will save the image to filename.
            If view is set to True will display the image."""
        nodes = self.get_cost_params()[1]
        tree_length = self.get_length()
        d_nodes = ((height - 40.0) / (2**(tree_length - 1))) 
        horz_step = ((width - 80) / tree_length)
        depth = 1
        window = pygame.display.set_mode((width, height))
        window.fill(WHITE)
        self.__drawTree(40, height / 2.0, window, tree_length, horz_step, depth, d_nodes)
        if save:
            pygame.image.save(window, filename)
        while view:
            for event in pygame.event.get():
                 if event.type == pygame.QUIT:
                     view = False
            pygame.display.flip()
        pygame.quit()
        return
            

def grow_tree(data, depth, max_depth = 500, Nmin = 5, labels = {}, start = False, feat_bag = False):
    """Function to grow a regression tree given some training data."""
    root = Tree(region_error(list(data.values())), numpy.mean(numpy.array(list(data.values()))), 
        numpy.std(numpy.array(list(data.values()))), start, len(data.values()))
    # regions has fewer than Nmin data points
    if (len(data.values()) <= Nmin):
        return root
    # length of tree exceeds max_depth
    if depth >= max_depth:
        return root
    num_vars = len(list(data.keys())[0])

    min_error = -1
    min_split = -1
    split_var = -1

    # Select variables to chose the split point from.
    # If feature bagging (for random forests) choose sqrt(p) variables
    # where p is the total number of variables.
    # Otherwise select all variables.
    if (feat_bag):
        cand_vars = random.sample(range(num_vars), int(num_vars**(0.5)))
    else:
        cand_vars = range(num_vars)
    # iterate over parameter space
    for i in cand_vars:
        var_space = [x[i] for x in data]
        if (min(var_space) == max(var_space)):
            continue
        # find optimal split point for parameter i
        split, error, ierr, numf = scipy.optimize.fminbound(error_function, min(var_space), max(var_space), args = (i, data), full_output = 1)
        # choose parameter that minimizes error
        if ((error < min_error) or (min_error == -1)):
            min_error = error
            min_split = split
            split_var = i
    # no more splits possible
    if split_var == -1:
        return root
    root.split_var = split_var
    root.split_val = min_split
    if split_var in labels:
        root.split_lab = labels[split_var]
    data1 = {}
    data2 = {}
    for i in data:
        if i[split_var] <= min_split:
            data1[i] = data[i]
        else:
            data2[i] = data[i]
    #grow right and left branches
    root.left = grow_tree(data1, depth + 1, max_depth = max_depth, Nmin = Nmin, labels = labels, feat_bag = feat_bag)
    root.right = grow_tree(data2, depth + 1, max_depth = max_depth, Nmin = Nmin, labels = labels, feat_bag = feat_bag)
    return root

def cvt(data, v, max_depth = 500, Nmin = 5, labels = {}):
    """Grows regression tree using v-fold cross validation.
    
    Data is a dictionary with elements of the form
    (x1, ..., xd) : y where x1, ..., xd are the parameter values and 
     y is the response value. 
     v is the number of folds for cross validation.
     max_depth is the maximum length of a branch emanating from the starting node.
     Nmin is the number of datapoints that must be present in a region to stop further partitions
     in that region. 
     labels is a dictionary where the keys are the indices for the parameters in the data
     and the values are strings assigning a label to the parameters. 
     See football_parserf.py for an example implementation."""
    full_tree = grow_tree(data, 0, max_depth = max_depth, Nmin = Nmin, 
        labels = labels, start = True)
    full_a, full_t = full_tree.prune_tree()

    # ak' = (ak * ak+1)^(1/2)
    a_s = []
    for i in range(len(full_a) - 1):
        a_s.append((full_a[i] * full_a[i+1])**(.5))
    a_s.append(full_a[-1])
    # stratify data
    pairs = sorted(data.items(), key=itemgetter(1))

    # break the data into v subsamples of roughly equal size
    lv_s = [dict(pairs[i::v]) for i in range(v)]
    # list of tree sequences for each training set
    t_vs = []
    # list of testing data for each training set
    test_vs = []
    # list of alpha values for each training set
    alpha_vs = []

    # grow and prune each training set
    for i in range(len(lv_s)):
        train = {k: v for d in lv_s[:i] for (k, v) in d.items()}
        train.update({k: v for d in lv_s[(i + 1):] for (k, v) in d.items()})
        test = lv_s[i]
        full_tree_v = grow_tree(train, 0, max_depth = max_depth, Nmin = Nmin, 
            labels = labels, start = True)  
        alphas_v, trees_v = full_tree_v.prune_tree()
        t_vs.append(trees_v)
        alpha_vs.append(alphas_v)
        test_vs.append(test)

    # choose the subtree that has the minimum cross-validated 
    # error estimate
    min_R = float("Inf")
    min_ind = 0
    for i in range(len(full_t)):
        ak = a_s[i]
        R_k = 0
        for j in range(len(t_vs)):
            # closest alpha value in sequence v to 
            # alphak'
            a_vs = alpha_vs[j]
            tr_vs = t_vs[j]
            alph_ind = bisect_right(a_vs, ak) - 1
            pairs = test_vs[j].items()
            para = [k[0] for k in pairs]
            va = [k[1] for k in pairs]
            pred_vals = tr_vs[alph_ind].predict_all(para)
            r_kv = numpy.sum((numpy.array(va) - numpy.array(pred_vals))**2)
            R_k = R_k + r_kv
        if (R_k < min_R):
            min_R = R_k
            min_ind = i
    return full_t[min_ind]

    

def error_function(split_point, split_var, data):
    """Function to minimize when choosing split point."""
    data1 = []
    data2 = []
    for i in data:
        if i[split_var] <= split_point:
            data1.append(data[i])
        else:
            data2.append(data[i])
    return region_error(data1) + region_error(data2)  
    
def region_error(data):
    """Calculates sum of squared error for some node in the regression tree."""
    data = numpy.array(list(data))
    return numpy.sum((data - numpy.mean(data))**2)



