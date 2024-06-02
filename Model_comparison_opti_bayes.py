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

import subprocess

# The command you want to execute
command = "pip list"

# Use subprocess to run the command
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
output, error = process.communicate()

# Print the output
if output:
    print("Output: " + output.decode())
if error:
    print("Error: " + error.decode())


# %% Catboost optimisation
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

def objective_function(learning_rate, depth, iterations):
    clf = CatBoostClassifier(
        cat_features=cat_features,
        learning_rate=learning_rate,
        depth=int(depth),
        iterations=int(iterations)
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
    return scores.mean()

pbounds = {
    'learning_rate': (0.01, 0.1),
    'depth': (4, 8),
    'iterations': (100, 300)
}

optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=42
)

print(optimizer.max)
# %% SVM optimisation
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

df = pd.read_csv('metabo_encoded.csv')

X = df.drop('MetabolicSyndrome', axis = 1)
y = df['MetabolicSyndrome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the objective function
def svm_cv(C, gamma):
    # Initialize the SVM model with the given parameters
    svm = SVC(C=C, gamma=gamma, kernel='rbf')

    # Compute the cross-validated score of the model
    score = cross_val_score(svm, X_train, y_train, cv=3).mean()

    return score

# Define the parameter bounds
pbounds = {
    'C': (0.1, 10.0),
    'gamma': (0.001, 1.0)
}

# Initialize the BayesianOptimization object
optimizer = BayesianOptimization(
    f=svm_cv,
    pbounds=pbounds,
    random_state=42
)

# Perform the optimization
optimizer.maximize(init_points=2, n_iter=50)

# Print the best parameters found
print("Best parameters found: ", optimizer.max['params'])
# %% XGBoost Optimisation


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

df = pd.read_csv('metabo_encoded.csv')

X = df.drop('MetabolicSyndrome', axis = 1)
y = df['MetabolicSyndrome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define the objective function
def xgb_cv(n_estimators, max_depth, gamma, min_child_weight, subsample, colsample_bytree):
    # Initialize the XGBoost model with the given parameters
    xgb = XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        gamma=gamma,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )

    # Compute the cross-validated score of the model
    score = cross_val_score(xgb, X_train, y_train, cv=3).mean()

    return score

# Define the parameter bounds
pbounds = {
    'n_estimators': (100, 500),
    'max_depth': (3, 10),
    'gamma': (0, 1),
    'min_child_weight': (0, 1),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}

# Initialize the BayesianOptimization object
optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=pbounds,
    random_state=42
)

# Perform the optimization
optimizer.maximize(init_points=2, n_iter=50)

# Print the best parameters found
print("Best parameters found: ", optimizer.max['params'])
# %%
