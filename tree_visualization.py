from sklearn.tree import export_graphviz
import six
import joblib

model = joblib.load('models/Random_model.sav')

dot_file = six.StringIO()
i_tree = 0
tree_num = model.estimators_
for tree_in_forest in model.estimators_:
    export_graphviz(tree_in_forest, out_file='output/random_forest/tree.dot', filled=True, rounded=True,
                    precision=2)
