import featuring
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

def bias_module(X):
    y = X['Survived']
    X = featuring.featuring(X)
    testX = pd.read_csv('static/test.csv')
    testX = featuring.featuring(testX)
    testY = pd.read_csv('static/gender_submission.csv')['Survived']

    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X, y)

    predY = lr.predict(testX)
    accuracy = accuracy_score(testY, predY)
    f1 = f1_score(testY, predY)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'F1 Score: {f1:.2f}')

X = pd.read_csv('static/train.csv')
bias_module(X)