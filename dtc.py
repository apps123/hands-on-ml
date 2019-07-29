from pandas import read_csv
from sklearn import tree
from decimal import Decimal
import pickle

data = read_csv("labeled.csv")
data.columns = ['purchases', 'mean', 'stddev', 'category']
# print(data)

predictors = ['purchases', 'mean', 'stddev']
X = data[predictors]
Y = data.category
decisionTreeClassifier = tree.DecisionTreeClassifier(criterion="entropy")
dTree = decisionTreeClassifier.fit(X, Y)

# TODO: calculate gini

# dotData = tree.export_graphviz(dTree, out_file=None)
# print(dotData)

# SAVE MODEL
filename = 'dtc.mlmodel'
pickle.dump(dTree, open(filename, 'wb'))
print('Model saved as file:' + filename)

# model = pickle.load(open(filename, 'rb'))
# print(model.predict([[128,0.1,0.222]]))

