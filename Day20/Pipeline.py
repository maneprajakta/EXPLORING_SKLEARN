'''
Pipeline
Steps involved in making Pipeline.
1.import pipeline from sklearn.
2.Define Pipeline.
3.call fit function on Pipeline.
4.call score function to check score.
'''

#import libraries needed

import pandas as pd 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

#read dataset 
df = pd.read_csv('pima.csv') 

#feature 
X = df.values[:,0:7] 
#prediction
Y = df.values[:,8] 

#splitting dataset 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7) 

#creating object of pipeline class
pipe  = Pipeline([('sc',StandardScaler()),('rfcl',RandomForestClassifier())])

#fitting on data
pipe.fit(X_train,y_train)
#predicting score 

print(pipe.score(X_test, y_test)

#also read some bit manipulation tricks 
#have written this in notes hope it does show error when compiled on lp
