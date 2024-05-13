import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

def categorize(name, new_titles):
    for new_title in new_titles:
        if new_title in str(name):
            return new_title
    return np.nan

def categorize_title(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex']=='Male':
            return "Mr"
        else:
            return "Mrs"
    else:
        return title

def featuring(X):
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']
    X['Title']=X['Name'].map(lambda x: categorize(x, title_list))
    X['Title']=X.apply(categorize_title, axis=1)

    # X=X.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    X=X.drop(['Name'], axis=1)

    X['Sex']=label_encoder.fit_transform(X['Sex'])
    X['Embarked']=label_encoder.fit_transform(X['Embarked'])
    X['Title']=label_encoder.fit_transform(X['Title'])

    columns_to_drop = ['PassengerId', 'Ticket', 'Cabin', 'Survived']
    X.drop(columns=[col for col in columns_to_drop if col in X], inplace=True)

    X = X.fillna(X.mean())

    return X