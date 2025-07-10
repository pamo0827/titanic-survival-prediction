import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os

train_data = pd.read_csv(r'C:\Users\yoshi\taitanic\train.csv')
test_data = pd.read_csv(r'C:\Users\yoshi\taitanic\test.csv')

os.makedirs(r'C:\Users\yoshi\taitanic\processed_data', exist_ok=True)

test_data['Perished'] = np.nan
combined_data = pd.concat([train_data, test_data], axis=0)

age_imputer = SimpleImputer(strategy='median')
combined_data['Age'] = age_imputer.fit_transform(combined_data[['Age']])

combined_data['Embarked'].fillna(combined_data['Embarked'].mode()[0], inplace=True)

combined_data['Fare'] = combined_data['Fare'].fillna(combined_data['Fare'].median())

combined_data['HasCabin'] = combined_data['Cabin'].notna().astype(int)

combined_data['Sex'] = combined_data['Sex'].map({'male': 0, 'female': 1})

embarked_dummies = pd.get_dummies(combined_data['Embarked'], prefix='Embarked', drop_first=True)
combined_data = pd.concat([combined_data, embarked_dummies], axis=1)

combined_data['FamilySize'] = combined_data['SibSp'] + combined_data['Parch'] + 1

combined_data['IsAlone'] = (combined_data['FamilySize'] == 1).astype(int)

combined_data['AgeGroup'] = pd.cut(combined_data['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])

combined_data['FareGroup'] = pd.qcut(combined_data['Fare'], 4, labels=[0, 1, 2, 3])

combined_data['Title'] = combined_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {
    "Mr": 0,
    "Miss": 1,
    "Mrs": 1,
    "Master": 2,
    "Dr": 3,
    "Rev": 3,
    "Col": 3,
    "Major": 3,
    "Mlle": 1,
    "Countess": 1,
    "Ms": 1,
    "Lady": 1,
    "Jonkheer": 3,
    "Don": 3,
    "Dona": 1,
    "Mme": 1,
    "Capt": 3,
    "Sir": 3
}
combined_data['Title'] = combined_data['Title'].map(title_mapping)
combined_data['Title'].fillna(0, inplace=True)

features_to_drop = ['Name', 'Ticket', 'Cabin', 'Embarked']
combined_data.drop(features_to_drop, axis=1, inplace=True)

train_processed = combined_data[:len(train_data)]
test_processed = combined_data[len(train_data):]
test_processed.drop('Perished', axis=1, inplace=True)

train_processed.to_csv(r'C:\Users\yoshi\taitanic\processed_data\train_processed.csv', index=False)
test_processed.to_csv(r'C:\Users\yoshi\taitanic\processed_data\test_processed.csv', index=False)
