import pandas as pd
import matplotlib.pyplot as plt


data_train = pd.read_csv('train.csv')


fig=plt.figure()
fig.set(alpha=0.65)
plt.title('Correlation Statistics by Gender')

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels(['Saved', 'Dead'], rotation=0)
ax1.legend(['Women/High Level'], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels(['Saved', 'Dead'], rotation=0)
plt.legend(['Women/Low Level'], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels(['Saved', 'Dead'], rotation=0)
plt.legend(['Men/High Level'], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels(['Saved', 'Dead'], rotation=0)
plt.legend(['Men/High Level'], loc='best')

plt.show()