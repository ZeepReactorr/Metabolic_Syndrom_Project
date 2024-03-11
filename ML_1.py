import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#Automate the path changing to the directory where the .csv is
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

#Initialize the dictionnaries to set a numerical value to the colomn where there aren't
dico_sex = {'Male':0, 'Female':1}
dico_race = {'White':0, 'Asian':1, 'Black':2, 'MexAmerican':3, 'Hispanic':4, 'Other': 5}
col = ['Age', 'Sex', 'Income', 'Race', 'WaistCirc', 'BMI', 'Albuminuria', 'UrAlbCr', 'UricAcid', 'BloodGlucose', 'HDL', 'Triglycerides']

#Load the dataset
df = pd.read_csv('Metabolic Syndrome.csv', sep = ',')

#replace the data with the value corresponding in the dictionnaries
df['Sex'] = [dico_sex[i] for i in df['Sex']]
df['Race'] = [dico_race[i] for i in df['Race']]

#Drop the rows where there are no values
df = df.dropna()

#Initialize the two dataframe, X as the dataset, Y as the target values
X = df[['Age', 'Sex', 'Income', 'Race', 'WaistCirc', 'BMI', 'Albuminuria', 'UrAlbCr', 'UricAcid', 'BloodGlucose', 'HDL', 'Triglycerides']]
Y = df['MetabolicSyndrome']

col = ['Age', 'Sex', 'Income', 'Race', 'WaistCirc', 'BMI', 'Albuminuria', 'UrAlbCr', 'UricAcid', 'BloodGlucose', 'HDL', 'Triglycerides']

#Classification with Logistic regression
def LR():    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , random_state=42)

    classifier = LogisticRegression(max_iter=50000)
    classifier.fit(X_train, Y_train)
    
    print(classifier.score(X_train, Y_train), classifier.score(X_test, Y_test))    
LR()

#Classification with Random Forest
def RFC():
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , random_state=42)
    clf = RandomForestClassifier(max_depth = 5, random_state=0)
    clf.fit(X_train, Y_train)
    print(clf.score(X_train, Y_train), clf.score(X_test, Y_test))
RFC()

#Classification with Logistic regression using a cross validation method
def VC_LR():
    classifier = LogisticRegression(max_iter=50000)
    
    R_square = cross_val_score(classifier, X, Y, cv=7)
    print(np.mean(R_square))
VC_LR()