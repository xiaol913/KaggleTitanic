import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt


data_train = pd.read_csv('train.csv')

fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title('Saved (1=saved)')

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.title("Level")
plt.ylabel('# of ppl')

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.grid(b=True, which='major', axis='y')
plt.title('Saved of Age (1=saved)')
plt.ylabel('Age')


plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Pclass')
plt.legend(('1st', '2nd', '3rd'), loc='best')

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title('Embarked')
plt.ylabel('# of ppl')


plt.show()