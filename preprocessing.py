# %% [markdown]
# # Data Transformation

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# %% [markdown]
# 
# ## Data Preprocessing
# 
# Cleaning the Dataset.
# 
# ### Data Summarization

# %%
data = pd.read_csv("Metabolic Syndrome.csv", index_col="seqn")
data["MetabolicSyndrome"] = data["MetabolicSyndrome"].astype("bool")
categorical_variables = ["Sex", "Marital", "Race"]

# %% [markdown]
# ### Replace the missing values, Imputer
# 
# Replacing the missing values by the mean of the column. 
# 
# Warning : does not take in count the Metabolic Syndrom state of the patient

# %%
data_imputed = data

for col in data_imputed.select_dtypes(include="number").columns:
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer.fit(data_imputed[[col]])
    data_imputed[col] = imputer.transform(data_imputed[[col]])
data_imputed.head()

# %% [markdown]
# ### Transform continuous variables into categorical variable and create an ”hot one encoding” structure

# %%
albuminurie = {0: "zero", 1: "one", 2:"two"}
data_encoded = data_imputed.replace({"Albuminuria":albuminurie})
data_encoded = pd.get_dummies(data_encoded)
data_encoded.head()

# %% [markdown]
# ### Scale the continuous variables
# 
# Warning : does not take in count the multiple outliers

# %%
data_scaled = data_encoded
for col in data_scaled.select_dtypes(include="number").columns:
    scaler = StandardScaler()
    scaler.fit_transform(data_scaled[[col]])
data_scaled.head()

# %% [markdown]
# ## Data Saving
# 
# save the dataset into a csv file.

# %%
data_scaled.to_csv("metabo_encoded.csv", index = True)

# %%
data_scaled.isna().count()


