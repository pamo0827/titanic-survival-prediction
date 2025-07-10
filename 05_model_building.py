import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

plt.rcParams['font.family'] = 'Meiryo'

train_data = pd.read_csv(r'C:\Users\yoshi\taitanic\processed_data\train_processed.csv')

with open(r'C:\Users\yoshi\taitanic\processed_data\selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f.readlines()]

print("Selected features:")
print(selected_features)

X = train_data[selected_features]
y = train_data['Perished']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset sizes:")
print(f"Training data: {X_train.shape}")
print(f"Validation data: {X_val.shape}")

dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)

train_accuracy = dt_model.score(X_train, y_train)
val_accuracy = dt_model.score(X_val, y_val)

print("\nBasic decision tree model accuracy:")
print(f"Training data accuracy: {train_accuracy:.4f}")
print(f"Validation data accuracy: {val_accuracy:.4f}")

param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest hyperparameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

best_train_accuracy = best_model.score(X_train, y_train)
best_val_accuracy = best_model.score(X_val, y_val)

print("\nOptimized decision tree model accuracy:")
print(f"Training data accuracy: {best_train_accuracy:.4f}")
print(f"Validation data accuracy: {best_val_accuracy:.4f}")

cv_scores = cross_val_score(best_model, X, y, cv=5)
print("\n5-fold cross-validation results:")
print(f"Mean accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

y_pred = best_model.predict(X_val)
conf_matrix = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['生存', '死亡'], yticklabels=['生存', '死亡'])
plt.xlabel('予測')
plt.ylabel('実際')
plt.title('混同行列')
plt.savefig(r'C:\Users\yoshi\taitanic\figures\confusion_matrix.png')
plt.close()

class_report = classification_report(y_val, y_pred, target_names=['Survived', 'Died'])
print("\nClassification report:")
print(class_report)

plt.figure(figsize=(20, 10))
plot_tree(best_model, feature_names=selected_features, class_names=['Survived', 'Died'], 
          filled=True, rounded=True, max_depth=3)
plt.savefig(r'C:\Users\yoshi\taitanic\figures\decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

import pickle
os.makedirs(r'C:\Users\yoshi\taitanic\models', exist_ok=True)
with open(r'C:\Users\yoshi\taitanic\models\best_decision_tree_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)