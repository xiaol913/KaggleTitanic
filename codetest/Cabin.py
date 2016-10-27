import pandas as pd

df = pd.read_csv('train.csv')

cabin = df[df.Cabin.notnull()][['PassengerId', 'Cabin']]
cb = cabin.values
newcb = []

for i in range(len(cb)):
    first = cb[i][1].split()
    if len(first) > 1:
        alpha = filter(str.isalpha, first[0])
        digit = filter(str.isdigit, first[0])
        if digit == '':
            digit = 0

        party = [cb[i][0], alpha, int(digit)]
        newcb.append(party)

    else:
        alpha = filter(str.isalpha, cb[i][1])
        digit = filter(str.isdigit, cb[i][1])
        if digit == '':
            digit = 0

        party = [cb[i][0], alpha, int(digit)]
        newcb.append(party)

df['Cabin_A'] = 0
df['Cabin_B'] = 0
df['Cabin_C'] = 0
df['Cabin_D'] = 0
df['Cabin_E'] = 0
df['Cabin_F'] = 0
df['Cabin_G'] = 0
df['Cabin_T'] = 0
df['Cabin_Num'] = 0

for j in range(len(newcb)):
    # num = newcb[j][2]
    # print num
    if newcb[j][1] == 'A':
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_A'] = 1
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_Num'] = newcb[j][2]
    elif newcb[j][1] == 'B':
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_B'] = 1
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_Num'] = newcb[j][2]
    elif newcb[j][1] == 'C':
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_C'] = 1
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_Num'] = newcb[j][2]
    elif newcb[j][1] == 'D':
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_D'] = 1
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_Num'] = newcb[j][2]
    elif newcb[j][1] == 'E':
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_E'] = 1
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_Num'] = newcb[j][2]
    elif newcb[j][1] == 'F':
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_F'] = 1
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_Num'] = newcb[j][2]
    elif newcb[j][1] == 'G':
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_G'] = 1
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_Num'] = newcb[j][2]
    elif newcb[j][1] == 'T':
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_T'] = 1
        df.loc[df.PassengerId == newcb[j][0], 'Cabin_Num'] = newcb[j][2]


# print df.dtypes