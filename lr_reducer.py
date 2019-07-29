# lr reducer on grid

import sys
import pickle
import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

dir_root = './datascience/'
filename = dir_root + 'models/lr.mlmodel'
model = pickle.load(open(filename, 'rb'))
filename = dir_root + 'models/orders.vectorizer'
vectorizer = pickle.load(open(filename, 'rb'))

def process():
    for line in sys.stdin:
        line = line.strip()
        cols = line.split('\t')
        if len(cols) < 4:
            continue
        (uid, did, subject, etime) = (cols[0], cols[1], cols[2], cols[3])
        if subject != None and subject != '':
            X_test = vectorizer.transform([subject])
            prediction = model.predict(X_test)
            print(uid + '\t' + did + '\t' + str(prediction[0]) + '\t' + etime)

if __name__ == "__main__":
    process()
