import pandas as pd
import numpy as np
import pickle
import os

test_data = pd.read_csv(r'C:\Users\yoshi\taitanic\processed_data\test_processed.csv')

original_test_data = pd.read_csv(r'C:\Users\yoshi\taitanic\test.csv')
passenger_ids = original_test_data['PassengerId']

with open(r'C:\Users\yoshi\taitanic\processed_data\selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f.readlines()]

print("Selected features:")
print(selected_features)

X_test = test_data[selected_features]

with open(r'C:\Users\yoshi\taitanic\models\best_decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Perished': predictions.astype(int)
})

os.makedirs(r'C:\Users\yoshi\taitanic\submission', exist_ok=True)

submission.to_csv(r'C:\Users\yoshi\taitanic\submission\titanic_predictions.csv', index=False)

print("Predictions on test data completed.")
print(r"Prediction results saved to C:\Users\yoshi\taitanic\submission\titanic_predictions.csv.")

print("\nPrediction statistics:")
print(f"Number of survivors: {(predictions == 0).sum()}")
print(f"Number of casualties: {(predictions == 1).sum()}")
print(f"Survival rate: {(predictions == 0).mean():.4f}")
