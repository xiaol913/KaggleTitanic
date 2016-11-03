import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn.preprocessing as preprocessing
import sys


def fix_missing_ages(df):
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    median_ages = np.zeros((2, 3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i, j] = df[(df['Sex'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna().median()

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j + 1), 'Age'] = median_ages[i, j]

    df['Child'] = 0
    df.loc[df.Age <= 12, 'Child'] = 1
    df['Sex_male'] = 0
    df['Sex_female'] = 0
    df.loc[df.Sex == 0, 'Sex_female'] = 1
    df.loc[df.Sex == 1, 'Sex_male'] = 1
    return df


def set_embarked_type(df):
    df['Embarked_S'] = 0
    df['Embarked_Q'] = 0
    df['Embarked_C'] = 0
    df.loc[df.Embarked == 'C', 'Embarked_C'] = 1
    df.loc[df.Embarked == 'S', 'Embarked_S'] = 1
    df.loc[df.Embarked == 'Q', 'Embarked_Q'] = 1
    return df


def set_pclass_type(df):
    df['Pclass_1'] = 0
    df['Pclass_2'] = 0
    df['Pclass_3'] = 0
    df.loc[df.Pclass == 1, 'Pclass_1'] = 1
    df.loc[df.Pclass == 2, 'Pclass_2'] = 1
    df.loc[df.Pclass == 3, 'Pclass_3'] = 1


def set_cabin_type(df):
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

    return df


def scale_age_fare(df):
    scaler = preprocessing.StandardScaler()
    df.loc[(df.Fare.isnull()), 'Fare'] = 0
    age_scale_param = scaler.fit(df['Age'])
    df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
    return df


def set_family_and_others(df):
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.Age * df.Pclass
    return df



def main(df):
    fix_missing_ages(df)
    set_embarked_type(df)
    set_cabin_type(df)
    scale_age_fare(df)
    set_family_and_others(df)
    set_pclass_type(df)
    return df


def train_test(train, test):
    # Load train.csv and test.csv
    df = pd.read_csv(train, header=0)
    data_test = pd.read_csv(test)
    # Train Data processing
    main(df)
    # Set Train Data feature
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|FamilySize|Child|Age*Class')
    train_np = train_df.values
    y = train_np[:, 0]
    x = train_np[:, 1:]
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(x, y)
    # Test Data processing
    main(data_test)
    # Set Test Data feature
    test = data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|FamilySize|Child|Age*Class')
    predictions = clf.predict(test)
    result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv('result.csv', index=False)



# train = 'train.csv'
# test = 'test.csv'
# train_test(train, test)
if __name__ == "__main__":
    # $python main.py train.csv test,csv
    train_test(sys.argv[1], sys.argv[2])

