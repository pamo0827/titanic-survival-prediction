import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

train_data = pd.read_csv(r'C:\Users\yoshi\taitanic\train.csv')
test_data = pd.read_csv(r'C:\Users\yoshi\taitanic\test.csv')

print("\nBasic information of training data:")
print("Data size:", train_data.shape)
print("\nFirst 5 rows of training data:")
print(train_data.head())
print("\nInformation about training data:")
print(train_data.info())
print("\nStatistical information of training data:")
print(train_data.describe())

print("\nMissing values in training data:")
print(train_data.isnull().sum())

print("\nDistribution of target variable (Perished):")
print(train_data['Perished'].value_counts())
print("Survival rate:", 1 - train_data['Perished'].mean())
