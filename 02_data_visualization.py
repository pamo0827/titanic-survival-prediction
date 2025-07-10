import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.family'] = 'Meiryo' 

train_data = pd.read_csv(r'C:\Users\yoshi\taitanic\train.csv')

os.makedirs(r'C:\Users\yoshi\taitanic\figures', exist_ok=True)

plt.figure(figsize=(10, 6))
sns.countplot(x='Perished', data=train_data)
plt.title('生存/死亡の分布')
plt.xlabel('死亡 (0=生存, 1=死亡)')
plt.ylabel('人数')
plt.savefig(r'C:\Users\yoshi\taitanic\figures\perished_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', hue='Perished', data=train_data)
plt.title('性別と生存/死亡の関係')
plt.xlabel('性別')
plt.ylabel('人数')
plt.legend(['生存', '死亡'])
plt.savefig(r'C:\Users\yoshi\taitanic\figures\sex_perished.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', hue='Perished', data=train_data)
plt.title('客室クラスと生存/死亡の関係')
plt.xlabel('客室クラス')
plt.ylabel('人数')
plt.legend(['生存', '死亡'])
plt.savefig(r'C:\Users\yoshi\taitanic\figures\pclass_perished.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(data=train_data, x='Age', hue='Perished', multiple='stack', bins=20)
plt.title('年齢分布と生存/死亡の関係')
plt.xlabel('年齢')
plt.ylabel('人数')
plt.legend(['死亡', '生存'])
plt.savefig(r'C:\Users\yoshi\taitanic\figures\age_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(data=train_data, x='Fare', hue='Perished', multiple='stack', bins=20)
plt.title('運賃分布と生存/死亡の関係')
plt.xlabel('運賃')
plt.ylabel('人数')
plt.legend(['死亡', '生存'])
plt.savefig(r'C:\Users\yoshi\taitanic\figures\fare_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='Embarked', hue='Perished', data=train_data)
plt.title('乗船港と生存/死亡の関係')
plt.xlabel('乗船港')
plt.ylabel('人数')
plt.legend(['生存', '死亡'])
plt.savefig(r'C:\Users\yoshi\taitanic\figures\embarked_perished.png')
plt.close()

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
plt.figure(figsize=(10, 6))
sns.countplot(x='FamilySize', hue='Perished', data=train_data)
plt.title('家族サイズと生存/死亡の関係')
plt.xlabel('家族サイズ')
plt.ylabel('人数')
plt.legend(['生存', '死亡'])
plt.savefig(r'C:\Users\yoshi\taitanic\figures\family_size_perished.png')
plt.close()

numeric_features = ['Perished', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
plt.figure(figsize=(12, 10))
correlation_matrix = train_data[numeric_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('数値特徴量の相関行列')
plt.savefig(r'C:\Users\yoshi\taitanic\figures\correlation_matrix.png')
plt.close()
