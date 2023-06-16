

import numpy as np
import pandas as pd
import pickle

train = pd.read_csv('train.csv')

train.describe()

train.isnull().sum()


train['Age'].fillna(train['Age'].mean(),inplace=True)
train['Embarked'].fillna(method='bfill',inplace = True)

train.drop('Cabin',axis=1,inplace=True)

train.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)


train['Embarked'].replace(['S','C','Q'] , [1,2,3], inplace= True)
train['Sex'].replace(['female','male'] , [0,1], inplace= True)

x_train = train.iloc[:,1:8].values
y_train = train.iloc[:,0].values

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)
lr.predict([[3,1,22,1,0,7.829,1]])[0]
pickle.dump(lr ,open('model.pkl', 'wb'))
