import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
train_data = np.load('train_data.npy', allow_pickle=True).item()
test_data = np.load('test_data.npy', allow_pickle=True).item()
X_train, y_train = train_data['X'], train_data['y']
X_test, y_test = test_data['X'], test_data['y']

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = {}

for kernel in kernels:
    model = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[kernel] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'y_pred': y_pred
    }

# Output results table
print("="*60)
print("SVM KERNEL COMPARISON")
print("="*60)
print(f"{'Kernel':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("-"*60)
for kernel in kernels:
    acc = results[kernel]['accuracy']
    prec = results[kernel]['precision']
    rec = results[kernel]['recall']
    f1 = results[kernel]['f1']
    print(f"{kernel:<10} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")

# Generate accuracy visualization (only errors)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('SVM Kernel Accuracy Visualization (Errors)', fontsize=16)

for i, kernel in enumerate(kernels):
    y_pred = results[kernel]['y_pred']
    acc = results[kernel]['accuracy']
    errors = (y_test != y_pred).astype(int)
    
    row, col = divmod(i, 2)
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=errors, cmap='RdYlGn_r', s=30, alpha=0.6)
    ax.set_title(f'SVM Test Set - {kernel.upper()}')
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')

plt.tight_layout()
plt.show()
