"""
one hot encoder 

author : Cloclochette
"""

# import
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

metabo =pd.read_csv("Metabolic Syndrome.csv",index_col='seqn')

# Data preprocessing

# Imputer

for i in metabo.select_dtypes(include="number").columns:
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    metabo[i] = imputer.fit_transform(metabo[[i]])
    
    
# LabelEncoder, convert categorical data to numerical data

for i in metabo.select_dtypes(exclude='number').columns:
    encoded_column = LabelEncoder().fit_transform(metabo[i])
    metabo[i] = encoded_column


# OneHotEncoder, convert categorical data to one-hot encoded data
for i in metabo.select_dtypes(exclude='number').columns:
    onehot_encoder = OneHotEncoder(sparse=False)
    encoded_column = onehot_encoder.fit_transform(metabo[[i]])
    encoded_df = pd.DataFrame(encoded_column, columns=onehot_encoder.get_feature_names_out([i]))
    metabo = pd.concat([metabo, encoded_df], axis=1)
    
    # Normalization
    scaler = MinMaxScaler()
    metabo_normalized = scaler.fit_transform(metabo.select_dtypes(include="number"))
    metabo_normalized = pd.DataFrame(metabo_normalized, columns=metabo.select_dtypes(include="number").columns)
    metabo = pd.concat([metabo.select_dtypes(exclude="number"), metabo_normalized], axis=1)

    
    

metabo.to_csv("metabo_encoded.csv", index=True)