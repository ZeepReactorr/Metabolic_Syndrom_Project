# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("metabo_encoded.csv")

# %%
df = df.dropna()
df = df.drop("seqn", axis=1)

# %%
for col in df.columns:
    if df[col].isnull().any() == True:
        df[col].fillna(df[col].mean(),inplace=True)

df.isnull().sum()

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df.drop('MetabolicSyndrome', axis=1)
Y = df["MetabolicSyndrome"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

RFC = RandomForestClassifier(n_estimators=100)
RFC.fit(X_train, Y_train)
test_prediction = RFC.predict(X_test)
train_prediction = RFC.predict(X_train)


print(f"Train score = {accuracy_score(train_prediction, Y_train )}\n\
Test score = {accuracy_score(test_prediction, Y_test)}")

# %%
FI = pd.DataFrame(RFC.feature_importances_, index = X_train.columns, columns=["importance"]).sort_values("importance", ascending=False)

dico = dict(zip(X.columns, RFC.feature_importances_))

dico = {k: v for k, v in sorted(dico.items(), key=lambda item: item[1])}
print(dico)
plt.barh(list(dico.keys()), list(dico.values()))


# %% [markdown]
# # test Catboost

# %%
cat_features = list(X.columns)
print(cat_features)

# %%
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8]
}

clf = CatBoostClassifier()

k_values = [3, 5, 10]

for k in k_values:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X, Y)

    # Print the best parameters found
    print(f"Best parameters found for {k}-fold CV: ", grid_search.best_params_)

    # Evaluate model on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)
    accuracy = accuracy_score(Y, y_pred)
    print(f"Accuracy with {k}-fold CV: ", accuracy)

# %% [markdown]
# # XBG classifying

# %%
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8]
}

xgb = XGBClassifier()

k_values = [3, 5, 10]

for k in k_values:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    grid_search = GridSearchCV(xgb, param_grid, cv=skf, scoring='accuracy')
    grid_search.fit(X, Y)

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
XGB_bp = XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=200)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

XGB_bp.fit(X_train, y_train)
y_pred_train = XGB_bp.predict(X_train)
y_pred_test = XGB_bp.predict(X_test)

print(f"Training accuracy = {accuracy_score(y_train, y_pred_train)}\n\
Testing accuracy = {accuracy_score(y_test, y_pred_test)}")

# %%
from sklearn.model_selection import cross_val_score, cross_validate

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

score_cvc = cross_validate(XGB_bp, X, Y, cv=skf)

print(np.mean(score_cvc))

# %% [markdown]
# # SVM

# %%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
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


# %% [markdown]
# # without unimportant features : 

# %%
X_trimmed = X.drop(['Marital_Divorced', 'Marital_Married', 'Marital_Separated', 'Marital_Single', 'Marital_Widowed',
            'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_MexAmerican', 'Race_Other', 'Race_White', 
            'Albuminuria_one', 'Albuminuria_two', 'Albuminuria_zero'], axis=1)

param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8]
}

clf = CatBoostClassifier()

k_values = [3, 5, 10]

for k in k_values:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_trimmed, Y)

    # Print the best parameters found
    print(f"Best parameters found for {k}-fold CV: ", grid_search.best_params_)

    # Evaluate model on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_trimmed)
    accuracy = accuracy_score(Y, y_pred)
    print(f"Accuracy with {k}-fold CV: ", accuracy)

# %%
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8]
}

xgb = XGBClassifier()

k_values = [3, 5, 10]

for k in k_values:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    grid_search = GridSearchCV(xgb, param_grid, cv=skf, scoring='accuracy')
    grid_search.fit(X, Y)

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

X_train, X_test, y_train, y_test = train_test_split(X_trimmed, Y, test_size=0.2, random_state=42)

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



