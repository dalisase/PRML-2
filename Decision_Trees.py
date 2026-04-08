import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
train_data = np.load('train_data.npy', allow_pickle=True).item()
test_data = np.load('test_data.npy', allow_pickle=True).item()
X_train, y_train = train_data['X'], train_data['y']
X_test, y_test = test_data['X'], test_data['y']

# Train model
model = DecisionTreeClassifier(random_state=42, max_depth=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print metrics
print("="*40)
print("RESULTS")
print("="*40)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print("\nFeature Importance:")
for i, imp in enumerate(model.feature_importances_):
    print(f"  {['X', 'Y', 'Z'][i]}: {imp:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualization
fig = plt.figure(figsize=(15, 5))
for idx, (data, title) in enumerate([(y_test, 'True'), (y_pred, 'Predicted'), 
                                      ((y_test != y_pred).astype(int), 'Decision Trees')]):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    cmap = 'RdYlGn_r' if idx == 2 else 'viridis'
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=data, cmap=cmap, s=50, alpha=0.7)
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    ax.set_title(f'Test Set - {title}')

plt.tight_layout()
plt.show()
