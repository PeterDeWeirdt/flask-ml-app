import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

if __name__ == '__main__':
    # create df
    train = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv') # change file path
    # drop null values
    train.dropna(inplace=True)
    # features and target
    target = 'survived'
    features = ['pclass', 'age', 'sibsp', 'fare']
    # X matrix, y vector
    X = train[features]
    y = train[target]
    # model
    model = LogisticRegression()
    model.fit(X, y)
    model.score(X, y)
    pickle.dump(model, open('model.pkl', 'wb'))
