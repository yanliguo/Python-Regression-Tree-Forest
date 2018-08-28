import csv
from make_tree_cart import *


def test_and_print(test_dict_lines, tree_root):
    errors = []
    for name in test_dict_lines.keys():
        value = test_dict_lines[name]
        features = value[0]
        predict, path = tree_root.lookup_with_path(features)
        path.reverse()
        act_val = value[1]
        print("\nFor test example: ", name, "\n\tlookup features:", path)
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

# number of bootstrap samples
B = 50
tree_root = make_cart_tree(train_dict, B, max_depth=500, Nmin=5, labels=labels)

test_and_print(test_dict_lines, tree_root)
tree_root.prune_cart_tree(test_dict)
test_and_print(test_dict_lines, tree_root)
