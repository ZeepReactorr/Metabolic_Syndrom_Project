# Metabolic_Syndrom_Project

This project is made by ZeepReactor, Cloclochette38, LinaKhlft and Alicia. \\ 

For this project a dataset from [Kaggle](https://www.kaggle.com/datasets/antimoni/metabolic-syndrome) was used.



## EDA 

For the Exploratory Data Analysis, we permformed various graphs, availables in the `eda.py` file.

## Preprocessing

The dataset was imputed with mean of the variable when data were missing. Then all the categorical data were encoded with a `OneHotEncoder` of scikit-learn. Finally, all the data were put to the same scale with a standardscaler. 

## Processing 

We trained various algorithms to search for the best one. We tried XGBoost, SVC and CatBoost. 
