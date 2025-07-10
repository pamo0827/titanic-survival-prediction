import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import os

plt.rcParams['font.family'] = 'Meiryo'

train_data = pd.read_csv(r'C:\Users\yoshi\taitanic\processed_data\train_processed.csv')

X = train_data.drop('Perished', axis=1)
y = train_data['Perished']

X = X.drop('PassengerId', axis=1)

dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X, y)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("Decision tree feature importance:")
print(feature_importance)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('決定木の特徴量重要度')
plt.tight_layout()
plt.savefig(r'C:\Users\yoshi\taitanic\figures\feature_importance.png')
plt.close()

mutual_info = mutual_info_classif(X, y, random_state=42)
mutual_info_df = pd.DataFrame({
    'Feature': X.columns,
    'Mutual_Information': mutual_info
})
mutual_info_df = mutual_info_df.sort_values('Mutual_Information', ascending=False)

print("\nMutual information (information gain):")
print(mutual_info_df)

plt.figure(figsize=(12, 8))
sns.barplot(x='Mutual_Information', y='Feature', data=mutual_info_df)
plt.title('特徴量の相互情報量（情報利得）')
plt.tight_layout()
plt.savefig(r'C:\Users\yoshi\taitanic\figures\mutual_information.png')
plt.close()

correlation = X.corrwith(y)
correlation_df = pd.DataFrame({
    'Feature': X.columns,
    'Correlation': correlation
})
correlation_df = correlation_df.sort_values('Correlation', ascending=False)

print("\nCorrelation with target variable:")
print(correlation_df)

plt.figure(figsize=(12, 8))
sns.barplot(x='Correlation', y='Feature', data=correlation_df)
plt.title('特徴量と目的変数の相関')
plt.tight_layout()
plt.savefig(r'C:\Users\yoshi\taitanic\figures\correlation_with_target.png')
plt.close()

top_features = feature_importance.head(10)['Feature'].tolist()
print("\nSelected important features (top 10):")
print(top_features)

with open(r'C:\Users\yoshi\taitanic\processed_data\selected_features.txt', 'w') as f:
    for feature in top_features:
        f.write(f"{feature}\n")
