### Importations des library 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import seaborn as sns

### Importation des datas
df = pd.read_csv(r"C:\Users\Alicia\Documents\BT4\ESME\Projet\Base_de_donnee\archive\Metabolic Syndrome.csv")
df = pd.DataFrame(df)



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



### Scatter Plot: Glucose dans le sang vs HDL

# Création du scatter plot
sns.scatterplot(data =df, x="BloodGlucose", y="HDL", hue = "MetabolicSyndrome",
                palette="Set2", edgecolor="k", s=100, alpha = 0.2)

# Définir les étiquettes des axes et le titre
plt.xlabel("Glucose dans le sang")
plt.ylabel("HDL")
plt.title("Scatter Plot: Glucose dans le sang vs HDL")

# Afficher le plot
plt.show()


### barplot: Revenus par race avec coloration selon le syndrome métabolique

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Race', y='Income', hue='MetabolicSyndrome', palette={1: 'red', 0: 'black'})

# Titre et labels
plt.title('Revenus par race avec coloration selon le syndrome métabolique')
plt.xlabel('Race')
plt.ylabel('Revenus')
plt.xticks(rotation=45)

# Légende
plt.legend(title='Syndrome métabolique')

# Affichage du barplot
plt.tight_layout()
plt.show()


### Barplot: Syndrome metabolique par race

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Race', y='MetabolicSyndrome')

# Titre et labels
plt.title('Syndrome metabolique par race')
plt.xlabel('Race')
plt.ylabel('Syndrome metabolique')

# Affichage du barplot
plt.tight_layout()
plt.show()

### Barplot: Syndrome metabolique par situation conjugal

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Marital', y='MetabolicSyndrome')

# Titre et labels
plt.title('Syndrome metabolique par situation conjugal')
plt.xlabel('Situation conjugal')
plt.ylabel('Syndrome metabolique')

# Affichage du barplot
plt.tight_layout()
plt.show()



### Barplot: Syndrome metabolique par race et situation conjugal
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Race', y='MetabolicSyndrome', hue = 'Marital',ci=None)

# Titre et labels
plt.title('Syndrome metabolique par race et situation conjugal')
plt.xlabel('Race')
plt.ylabel('Syndrome metabolique')

# Affichage du barplot
plt.tight_layout()
plt.show()


















