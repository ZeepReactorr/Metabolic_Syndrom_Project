# Metabolic_Syndrom_Project

This project is made by ZeepReactor, Cloclochette38, LinaKhlft and Alicia.

The objective was to determine the impact of integrating new societal and physiological factors on the robustness of a model predicting metabolic syndrome. To do so, a dataset from [Kaggle](https://www.kaggle.com/datasets/antimoni/metabolic-syndrome) was used.

## EDA 

For the Exploratory Data Analysis, we permformed various graphs, availables in the [eda.py](https://github.com/ZeepReactorr/Metabolic_Syndrom_Project/blob/main/eda.py) file. histograms, bar charts, and heatmaps were used to examine the dataset's structure and identify patterns.
The dataset does not have an overrepresentation of any particular population. **Obesity** and **diabetes** emerge as key factors, indicating that these variables play a significant role in the analysis, potentially influencing the outcomes or predictions of the model.

## Preprocessing

The dataset was imputed with mean of variables when data were missing. Then, all the categorical data were encoded with a `OneHotEncoder` of scikit-learn. Finally, all the data were put to the same scale with a standardscaler.

Using `feature_importances`, only variables having a relative importance greater than or equal to 0.018 were retained, available in the [feature_selection.py](https://github.com/ZeepReactorr/Metabolic_Syndrom_Project/blob/main/feature_selection.py) file.

## Processing 

We trained various algorithms to search for the best one in the [model_comparison.py](https://github.com/ZeepReactorr/Metabolic_Syndrom_Project/blob/main/model_comparison.py) file. We tried XGBoost, SVC and CatBoost.

**XGBoost** was selected as the final model because of its strong performance with data and its potential for further improvement. The computation time was optimized using `Bayesian optimization` method, availables in [Model_comparison_opti_bayes.py](https://github.com/ZeepReactorr/Metabolic_Syndrom_Project/blob/main/Model_comparison_opti_bayes.py) file.
