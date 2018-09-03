import csv
import sys
from make_tree_cart import *
from plot import plot_cart, plot_feature_importance
from model_io import *


def test_and_print(test_dict_lines, tree_root, label_dicts=None):
    errors = []
    for name in test_dict_lines.keys():
        value = test_dict_lines[name]
        features = value[0]
        predict, path = tree_root.lookup_with_path(features)
        path.reverse()
        act_val = value[1]
        # print("\nFor test example: ", name, "\n\tlookup features:", path)
        feature_counts, feature_percents = feature_weights(path)
        if label_dicts:
            # converts feature index to feature name
            feature_counts = {label_dicts.get(k, ''): v for (k, v) in feature_counts.items()}
        print("\nFor test example: ", name, "\n\tlookup features:", feature_counts)
        print("\tActual value:", act_val, "\tpredicted value:", predict)
        errors.append(abs(act_val - predict))
    print("** Avg error:", numpy.mean(numpy.array(errors)))
        

football = open("football.csv", "r")
f_reader = csv.DictReader(football)
years = ["VII", "VIII", "IX", "X", "XI", "XII", "XIII"]
variables = ["Age", "Att", "YA", "Rec", "YR", "RRTD", "Fmb"]
labels = {0: "Age", 1: "Att", 2: "YA", 3: "Rec", 4: "YR", 5: "RRTD", 6: "Fmb"}
train_dict = {}
test_dict = {}
test_dict_lines = {}
test_year = "XII"

# Construct a dictionary with entries of the form
# (x1, ..., xd) : y where x1, ..., xd are the parameter values and 
#  y is the response value.
for row in f_reader:
    for i in range(len(years) - 1):
        if row[years[i] + "Att"] == "":
            continue
        dat = []
        for var in variables:
            if row[years[i] + var] == "":
                dat.append(0)
            else:
                dat.append(float(row[years[i] + var]))
        res = row[years[i + 1] + "Fantasy"]
        if res == "":
            res = 0
        if years[i] == test_year:
            test_dict[tuple(dat)] = float(res)
            test_dict_lines[row["Name"]] = [tuple(dat), float(res)]
        else:
            train_dict[tuple(dat)] = float(res)

args = sys.argv.copy()

# number of bootstrap samples
B = 50

# CMD: "--load file" loads model from file
load_idx = args.index('--load') if '--load' in args else -1
loaded = False
tree_root = None
if load_idx >= 0:
    model_file_name = args[load_idx + 1]
    tree_root = load_model(model_file_name)
    loaded = tree_root is not None
    print('loaded from', model_file_name)
if tree_root is None:
    tree_root = make_cart_tree(train_dict, B, max_depth=500, Nmin=5, labels=labels)

# CMD: "--test" tells the app to run test cases
if '--test' in args:
    test_and_print(test_dict_lines, tree_root, label_dicts=labels)
    
# if the model is load from file,
# no need to prune again
if not loaded:
    tree_root.prune_cart_tree(test_dict)

if '--test' in args:
    test_and_print(test_dict_lines, tree_root)


# CMD:  "--save file" will save model to 'file'
if '--save' in args:
    save_idx = args.index('--save')
    save_file_name = args[save_idx + 1]
    save_model(save_file_name, tree_root)


# CMD: '--plot-tree' will plot the entire tree
if '--plot-tree' in args:
    plot_cart(tree_root)

# CMD: '--plot-weights' will plot tree split weights
if '--plot-weights' in args:
    wts = tree_root.get_feature_weights()
    wts = {labels.get(k, ''): v for (k, v) in wts.items()}
    plot_feature_importance(wts, title='Split feature weights')


# CMD: '--plot-feature' plots split counts for feature of the first test case
if '--plot-feature' in args:
    name = list(test_dict_lines.keys())[0]
    feature, actual = test_dict_lines[name]
    predict, path = tree_root.lookup_with_path(feature)
    path.reverse()
    feature_counts, feature_percents = feature_weights(path)
    # converts feature index to feature name
    feature_counts = {labels.get(k, ''): v for (k, v) in feature_counts.items()}
    plot_feature_importance(feature_counts)
