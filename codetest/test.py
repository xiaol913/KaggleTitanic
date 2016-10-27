import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame


data_train = pd.read_csv('train.csv')

fig=plt.figure()
fig.set(alpha=0.65)
plt.title('Correlation Statistics by Gender')

data_train.Survived[data_train.Age < 12].plot(kind='kde')
data_train.Survived[data_train.Age >= 12].plot(kind='kde')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Pclass')
plt.legend(('<12', '>=12'), loc='best')


plt.show()