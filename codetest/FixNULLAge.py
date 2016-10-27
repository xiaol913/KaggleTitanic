import pandas as pd
import numpy as np

# read csv
df = pd.read_csv('train.csv', header=0)
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)


median_ages = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df[(df['Sex'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j + 1), 'Age'] = median_ages[i, j]

