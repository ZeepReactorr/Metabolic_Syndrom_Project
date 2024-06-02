# %% [markdown]
# # **Metabolic Syndrome: Model Comparison and Evaluation**
# 
# Comparison of svm, XGboost and Catboost

# %%
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('Metabolic Syndrome.csv')

# %%
df.info()
df.isnull().sum()

# %%
df = df.dropna(subset=['Marital'])
for col in df.columns:
    if df[col].isnull().any() == True:
        df[col].fillna(df[col].mean(),inplace=True)

df.isnull().sum()

# %% [markdown]
# ### CatBoost

# %%
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

X = df.drop('MetabolicSyndrome', axis = 1)
y = df['MetabolicSyndrome']

param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8]
}

cat_features = ['Sex', 'Marital', 'Race']

clf = CatBoostClassifier(cat_features=cat_features)

k_values = [3, 5, 10]

for k in k_values:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X, y)

    # Print the best parameters found
    print(f"Best parameters found for {k}-fold CV: ", grid_search.best_params_)

    # Evaluate model on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy with {k}-fold CV: ", accuracy)

# %% [markdown]
# ### SVM & XGBoost

# %%
for col in df.columns:
    if df[col].isnull().any() == True:
        df[col].fillna(df[col].mean(),inplace=True)

# %%
from sklearn.preprocessing import LabelEncoder

for col in df.columns:
    if df[col].dtype != float:
        labelencoder = LabelEncoder()
        encode_col = labelencoder.fit_transform(df[col])
        df[col] = encode_col.tolist()
df.info()

# %%
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

X = df.drop('MetabolicSyndrome', axis = 1)
y = df['MetabolicSyndrome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8]
}

xgb = XGBClassifier()

k_values = [3, 5, 10]

for k in k_values:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X, y)

    # Print the best parameters found for this k
    print(f"Best parameters found for {k}-fold CV: ", grid_search.best_params_)
    
    # Evaluate model with best parameters on the training set
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Train accuracy with {k}-fold CV: ", train_accuracy)
    
    # Evaluate model with best parameters on the test set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy with {k}-fold CV: ", test_accuracy)

# %%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

svm = SVC()

k_values = [3, 5, 10]

for k in k_values:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=skf, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters found for this k
    print(f"Best parameters found for {k}-fold CV: ", grid_search.best_params_)
    
    # Evaluate model with best parameters on the training set
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Train accuracy with {k}-fold CV: ", train_accuracy)
    
    # Evaluate model with best parameters on the test set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy with {k}-fold CV: ", test_accuracy)



