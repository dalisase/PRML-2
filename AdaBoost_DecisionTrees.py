import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# MODIFIED: Added GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
train_data = np.load('train_data.npy', allow_pickle=True).item()
test_data = np.load('test_data.npy', allow_pickle=True).item()
X_train, y_train = train_data['X'], train_data['y']
X_test, y_test = test_data['X'], test_data['y']

# MODIFIED: Added hyperparameter tuning with GridSearchCV to improve accuracy
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3]
}

base_estimator = DecisionTreeClassifier(random_state=42)
model = AdaBoostClassifier(estimator=base_estimator, random_state=42)

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# MODIFIED: Use best model from grid search instead of fixed parameters
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print metrics
print("="*40)
print("ADABOOST + DECISION TREE RESULTS")
print("="*40)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")

# MODIFIED: Fixed reference to use best_model.estimators_ for feature importance
feature_importances = np.mean([estimator.feature_importances_ for estimator in best_model.estimators_], axis=0)
print("\nFeature Importance (Averaged):")
for i, imp in enumerate(feature_importances):
    print(f"  {['X', 'Y', 'Z'][i]}: {imp:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualization
fig = plt.figure(figsize=(15, 5))
for idx, (data, title) in enumerate([(y_test, 'True'), (y_pred, 'Predicted'), 
                                      ((y_test != y_pred).astype(int), 'AdaBoost + Decision Trees')]):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    cmap = 'RdYlGn_r' if idx == 2 else 'viridis'
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=data, cmap=cmap, s=50, alpha=0.7)
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    ax.set_title(f'AdaBoost Test Set - {title}')

plt.tight_layout()
plt.show()
