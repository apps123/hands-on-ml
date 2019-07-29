# reducer code

import sys
import pickle
import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

dir_root = './datascience/'
filename = dir_root + 'models/dtc.mlmodel'
model = pickle.load(open(filename, 'rb'))

def classify():
    for line in sys.stdin:
        line = line.strip()
        cols = line.split('\t')
        if len(cols) < 4:
            continue
        (uid, purchases, mean, stddev) = (cols[0], int(cols[1]), float(cols[2]), float(cols[3]))
        prediction = model.predict([[purchases, mean, stddev]])
        print(uid + '\t' + str(prediction[0]))

if __name__ == "__main__":
    classify()
