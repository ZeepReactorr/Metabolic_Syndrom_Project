# %% [markdown]
# # EDA Metabolic syndrom
# 
# Exploratory Data notebook for the dataset https://www.kaggle.com/datasets/antimoni/metabolic-syndrome/data
# 
# First we import the packages we will use for the exploration :
# - matplotlib
# - pandas
# - seaborn
# 
# What we want to see : 
# - If the dataset follows the society's general tendencies so that it can be compared properly
# - Check the correlation between featuempiric

# %%
from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd
import seaborn as sns
import numpy as np
from dataclasses import dataclass

df = pd.read_csv('Metabolic Syndrome.csv', sep = ',')

# %% [markdown]
# ## Some basics

# %%
df.info()

# %%
df.describe(include='O').T

# %%
def check_met(value):
    if value == 1:
        return 'Metabolic Syndrome'
    else:
        return 'Healthy'

df['Metabolic_Dist'] = df['MetabolicSyndrome'].apply(check_met)

# %%
df.hist(figsize=(15,12),bins = 15)
plt.title("Featuempiric Distribution")
plt.show()

# %%
#counts the number of patients healthy and with MS
sex_colors = {'Male':'#377eb8', 'Female':'#f781bf'}

target = df.MetabolicSyndrome.value_counts()

plt.bar(["Healthy", "Sick : MS"], [i for i in list(target)], color = ['tab:blue', 'tab:red'], label=[f"{np.round(target[0]/sum(list(target)), 3)}%", f"{np.round(target[1]/sum(list(target)),3)}%"])
plt.ylabel("Number of patients")
plt.legend()

# %%
df_sex_ages = df[["Age", "Sex"]]
df_male = df_sex_ages.loc[df_sex_ages.Sex=='Male']
df_female = df_sex_ages.loc[df_sex_ages.Sex=='Female']

bins = [0, 20, 30, 40, 50, 60, 70, 80, float('inf')]
labels = ['Under 20', '21 to 30', '31 to 40', '41 to 50', '51 to 60', '61 to 70', '71 to 80', 'Over 81']

df_male['AgeGroup'] = pd.cut(df_male['Age'], bins=bins, labels=labels, right=False)

pd.options.mode.chained_assignment = None  # default='warn'

ctrl_distribution_male = [142, 68, 71, 72, 77, 68, 55]
ctrl_distribution_male = [i*2.0645558144130888138670069474839 for i in ctrl_distribution_male]

bins = [0, 20, 30, 40, 50, 60, 70, 80, float('inf')]
labels = ['Under 20', '21 to 30', '31 to 40', '41 to 50', '51 to 60', '61 to 70', '71 to 80', 'Over 81']

df_female['AgeGroup'] = pd.cut(df_female['Age'], bins=bins, labels=labels, right=False)

pd.options.mode.chained_assignment = None  # default='warn'

ctrl_distribution_female = [141, 68, 74, 78, 81, 75, 69]
ctrl_distribution_female = [i*1.9393589198996768975264360179271 for i in ctrl_distribution_female]

age_group_counts_male = df_male['AgeGroup'].value_counts()
age_group_counts_female = df_female['AgeGroup'].value_counts()

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.pie(age_group_counts_male, labels=age_group_counts_male.index, autopct='%1.1f%%')
plt.title('Distribution of Age Groups for Males')

plt.subplot(1, 2, 2)
plt.pie(ctrl_distribution_male, autopct='%1.1f%%')
plt.title('Control Distribution for Males')

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.pie(age_group_counts_female, labels=age_group_counts_female.index, autopct='%1.1f%%')
plt.title('Distribution of Age Groups for Females')

plt.subplot(1, 2, 2)
plt.pie(ctrl_distribution_female, autopct='%1.1f%%')
plt.title('Control Distribution for Females')

plt.tight_layout()
plt.show()

# %%
def check_met(value):
    if value == 1:
        return 'Metabolic Syndrome'
    else:
        return 'Healthy'

df['Metabolic_Dist'] = df['MetabolicSyndrome'].apply(check_met)

# %%
df['Sex'] = df['Sex'].astype('string').fillna('NaN')
df['Marital'] = df['Marital'].astype('string').fillna('NaN')
df['Race'] = df['Race'].astype('string').fillna('NaN')

fig = px.sunburst(df, path = ['Sex', 'Marital', 'Race'], color = 'Sex', color_discrete_map = sex_colors)
fig.update_traces(textinfo = "label + value")

fig.update_layout(autosize=False, width=1000,height=800)

# %%
def kde_boxplot(df,df_col):
    for col in df_col:
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        
        sns.histplot(df, x = col, hue = 'Metabolic_Dist', kde = True, multiple="stack", ax=axes[0])
        
        sns.boxplot(df, y = 'Sex', x = col, hue = 'Metabolic_Dist', ax=axes[1])
        
        fig.suptitle(f'Numeric Feature : {col}', fontsize=16, fontweight='bold')
        fig.subplots_adjust(wspace=0.2)
        plt.show()

# %%
### Bar plot du metabolic syndrome par groupe d'age et sexe

# Création de catégories d'âges
bins = [20, 30, 40, 50, 60, 70, 81]
labels = ['20-29', '30-39', '40-49', '50-59', '60-69','70-80']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Calcul de la proportion de personnes atteintes du syndrome métabolique par groupe d'âge et sexe
grouped = df.groupby(['AgeGroup', 'Sex'])['MetabolicSyndrome'].mean().reset_index()

# Création du barplot
plt.figure(figsize=(10, 6))
sns.barplot(data=grouped, x='AgeGroup', y='MetabolicSyndrome', hue='Sex', palette= {'Male': 'lightblue', 'Female': 'pink'})
plt.title('Proportion de personnes avec syndrome métabolique par groupe d\'âge et sexe')
plt.xlabel('Catégorie d\'âge')
plt.ylabel('Proportion de personnes avec syndrome métabolique')

# Légende
plt.legend(title='Sexe')

# Affichage du barplot
plt.tight_layout()
plt.show()

# %%
df_col = df.drop(columns=["seqn",'Albuminuria','MetabolicSyndrome', 'Age']).select_dtypes("number")

kde_boxplot(df,df_col)

# %%
fig = px.box(df, x="Sex", y="BMI", facet_col='Metabolic_Dist', color = 'Sex', color_discrete_map = sex_colors)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(autosize=False, width=800,height=600)

# %%
plt.figure(figsize=(10,8), dpi= 80)
sns.pairplot(df.loc[:, [i for i in df.columns if i != "seqn"]], 
             kind="scatter", hue="MetabolicSyndrome", plot_kws=dict(palette="Set2", edgecolor="k", s=100, alpha = 0.2))
plt.show()

# %%
from scipy.stats import chi2_contingency

df_sorted = df[["Sex",  'Age', "Marital", "Income", "Race", 'WaistCirc', 'BMI', 'UrAlbCr', 'UricAcid', 'BloodGlucose', 'HDL', 'Triglycerides',]]

def df_maker(col, row):
    DF = pd.DataFrame(data = None, index=row, columns=col)
    return DF

df_sorted = df_maker(list(df_sorted.columns), ["Overall normal distribution"])

def chi2test(feature, featname):
    contingency_table = pd.crosstab(df[feature],df['MetabolicSyndrome'])

    empiric = chi2_contingency(contingency_table)
    pvalue = round(empiric[1], 4)

    lim = 0.05
    if pvalue < lim:
        return True
    else:
        return False

for col in df_sorted.columns:
    try :
        df_sorted[col] = [chi2test(col, col+"feature")]
    except Exception:
        pass

display(df_sorted)

# %%
df_matrix = df.drop(columns=["seqn",'Marital','Sex', 'Race'])
corr_matrix = df_matrix.corr().round(2)
plt.figure(figsize = (12,10))

sns.heatmap(corr_matrix, annot = True)

# %% [markdown]
# Metabolic Syndrom Conditions

# %%
df['Glycemia'] = df['BloodGlucose'].apply(lambda value: 'Hypoglycemia' if value < 70 else 'Normal' if 70 <= value < 110 else 'Hyperglycemia')

# %%
glycemia_colors = {'Hypoglycemia': '#377eb8', 'Normal': '#4daf4a', 'Hyperglycemia': '#e41a1c'}

fig = px.histogram(df, x = "BloodGlucose", facet_col = 'Metabolic_Dist', facet_row = 'Sex', color = 'Glycemia', color_discrete_map = glycemia_colors, log_y = True)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# %%
df['BMI_statut'] = df['BMI'].apply(lambda value: 'Underweight' if value < 18.5 else 'Normal' if 18.5 <= value < 25 else 'Overweight' if 25 <= value < 30 else 'Obese' if 30 <= value < 40 else 'Severely Obese' if 40 <= value < 50 else 'Hyper Obese')

# %%
fig = px.histogram(df, x = 'BMI', facet_col = 'Metabolic_Dist', facet_row = 'Sex', color = 'BMI_statut')

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# %%
df['Lipid'] = df.apply(lambda row: 'Normal' if (row['Sex'] == 'Female' and row['WaistCirc'] < 88) or (row['Sex'] == 'Male' and row['WaistCirc'] < 102) else 'Adipocytes_Excess', axis=1)

# %%
df['Lipid'].value_counts()

# %%
waist_colors = {'Normal': '#4daf4a', 'Adipocytes_Excess': '#e41a1c'}

fig = px.histogram(df, x = 'WaistCirc', facet_col = 'Metabolic_Dist', facet_row = 'Sex', color = 'Lipid', color_discrete_map = waist_colors)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# %%
df['Cholesterol'] = df.apply(lambda row: 'Normal' if (row['Sex'] == 'Female' and row['HDL'] >= 40) or (row['Sex'] == 'Male' and row['HDL'] >= 50) else 'Poor_HDL', axis=1)

# %%
HDL_colors = {'Normal': '#4daf4a', 'Poor_HDL': '#e41a1c'}

fig = px.histogram(df, x = 'HDL', facet_col = 'Metabolic_Dist', facet_row = 'Sex', color = 'Cholesterol', color_discrete_map = HDL_colors)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# %%
df['Trig_status'] = df['Triglycerides'].apply(lambda value: 'Normal' if value < 150 else 'Hypertriglyceridemia')

# %%
triglycerides_colors = {'Normal': '#4daf4a', 'Hypertriglyceridemia': '#e41a1c'}

fig = px.histogram(df, x = "Triglycerides", facet_col = 'Metabolic_Dist', facet_row = 'Sex', color = 'Trig_status', color_discrete_map = triglycerides_colors)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# %%
def check_metabolic_syndrome(row):
    conditions = 0
    if row['Glycemia'] == 'Hyperglycemia':
        conditions += 1
    if row['Lipid'] == 'Adipocytes_Excess':
        conditions += 1
    if row['Cholesterol'] == 'Poor_HDL':
        conditions += 1
    elif row['Triglycerides'] == 'Hypertriglyceridemia':
        conditions += 1
    
    return 'Metabolic Syndrome' if conditions >= 3 else 'Healthy'

df['met_conditions'] = df.apply(check_metabolic_syndrome, axis=1)


# %%
ctrl_distribution_met = df['met_conditions'].value_counts()

metabolic_synd = df['Metabolic_Dist'].value_counts()

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.pie(ctrl_distribution_met, labels=ctrl_distribution_met.index, autopct='%1.1f%%')
plt.title('Distribution of Metabolic Syndrome with conditions')

plt.subplot(1, 2, 2)
plt.pie(metabolic_synd, labels=metabolic_synd.index, autopct='%1.1f%%')
plt.title('Control Distribution of Metabolic Syndrome')


