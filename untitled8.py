import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import plotly


df = pd.read_csv(r"C:\Users\Alicia\Documents\BT4\ESME\Projet\Base_de_donnee\archive\Metabolic Syndrome.csv")
data_base = pd.read_csv(r"C:\Users\Alicia\Documents\BT4\ESME\Projet\Base_de_donnee\archive\Metabolic Syndrome.csv")


#Initialize the dictionnaries to set a numerical value to the colomn where there aren't
dico_sex = {'Male':0, 'Female':1}
dico_race = {'White':0, 'Asian':1, 'Black':2, 'MexAmerican':3, 'Hispanic':4, 'Other': 5}
col = ['Age', 'Sex', 'Income', 'Race', 'WaistCirc', 'BMI', 'Albuminuria', 'UrAlbCr', 'UricAcid', 'BloodGlucose', 'HDL', 'Triglycerides']

#replace the data with the value corresponding in the dictionnaries
df['Sex'] = [dico_sex[i] for i in df['Sex']]
df['Race'] = [dico_race[i] for i in df['Race']]


#Drop the rows where there are no values
df = df.dropna()
del df['Marital']

import plotly.express as px

px.scatter(data_base, x="Age", y="Sex")


