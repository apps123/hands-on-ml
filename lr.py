import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sklearn.pipeline
import numpy as np
import pickle

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('labeled.tsv', sep='\t', names=['text','is_order'])
df = df.dropna()
# print(df)

X_train, X_test, y_train, y_test = train_test_split(df.text, df.is_order, test_size=0.20, random_state=0)
test_df = X_test

# max_features=10-15% of the labeled set (eyeball) - for this domain
vectorizer = CountVectorizer(min_df=4, stop_words='english', max_features=2000)

# vectorizer = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word',
#                         stop_words= 'english',ngram_range=(1,3),dtype=np.float32)    
X_train = vectorizer.fit_transform(X_train)
# print(vectorizer.get_feature_names())
X_test = vectorizer.transform(X_test)

# LOGISTIC REGRESSION
lr = LogisticRegression(solver='lbfgs', multi_class='ovr')
lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)
# print(lr_prediction)

# METRICS
print("Accuracy  = " + str(accuracy_score(y_test, lr_prediction)*100)+"%")
print('F1 score  = ', f1_score(y_test, lr_prediction, average='weighted'))
print('Recall    = ', recall_score(y_test, lr_prediction, average='weighted'))
print('Precision = ', precision_score(y_test, lr_prediction, average='weighted'))
print('\n Clasification Report:\n', classification_report(y_test, lr_prediction))
print('\n confussion matrix:\n', confusion_matrix(y_test, lr_prediction))

# VISUALIZE RESULTS
lr_results = np.array(list(zip(test_df.iloc[0:],lr_prediction)))
lr_results = pd.DataFrame(lr_results, columns=['text', 'is_order'])
# print(lr_results)

# SAVE MODEL
filename = 'orders.mlmodel'
pickle.dump(lr, open(filename, 'wb'))
print('Model saved as file:' + filename)

# SAVE VECTORIZER
filename = 'orders.vectorizer'
pickle.dump(vectorizer, open(filename, 'wb'))
print('Vectorizer saved as file:' + filename)

filename = 'orders.mlmodel'
model = pickle.load(open(filename, 'rb'))
result = model.score(X_test, y_test)
print('Score from saved model:' + str(result))
