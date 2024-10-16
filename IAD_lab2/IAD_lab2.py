import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                             f1_score, precision_recall_curve, roc_curve, 
                             auc, classification_report)
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs

X, y_true = make_blobs (n_samples = 400, centers =4, cluster_std = 0.60, random_state = 0)
rng = np.random.RandomState (13)
X_stretched = np.dot(X, rng.randn (2 , 2))

plt.scatter(X_stretched[:, 0], X_stretched[:, 1], c=y_true, cmap='viridis', edgecolor='k')
plt.title('Dataset (a)')
plt.show()

X_train, X_val, y_train, y_val = train_test_split(X_stretched, y_true, test_size=0.2, random_state=42, stratify=y_true)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

linear_svc = LinearSVC(C=1e5, max_iter=10000, random_state=42)
svc_linear = SVC(kernel='linear', C=1e5, random_state=42)

linear_svc.fit(X_train_scaled, y_train)
svc_linear.fit(X_train_scaled, y_train)

linear_svc_small = LinearSVC(C=100, max_iter=10000, random_state=42)
svc_linear_small = SVC(kernel='linear', C=100, random_state=42)

linear_svc_small.fit(X_train_scaled, y_train)
svc_linear_small.fit(X_train_scaled, y_train)

def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
    plt.title(title)
    plt.show()

plot_decision_boundary(linear_svc, X_val_scaled, y_val, 'LinearSVC Decision Boundary')
plot_decision_boundary(svc_linear, X_val_scaled, y_val, 'SVC (linear kernel) Decision Boundary')

plot_decision_boundary(linear_svc_small, X_val_scaled, y_val, 'LinearSVC Decision Boundary (small)')
plot_decision_boundary(svc_linear_small, X_val_scaled, y_val, 'SVC (linear kernel) Decision Boundary (small)')

degrees = [2, 3, 4]
coef0_values = [0, 1, 10]

svc_poly_models = {}

for degree in degrees:
    for coef0 in coef0_values:
        model_name = f'SVC_poly_degree_{degree}_coef0_{coef0}'
        svc_poly = SVC(kernel='poly', degree=degree, coef0=coef0, C=1.0, random_state=42)
        svc_poly.fit(X_train_scaled, y_train)
        svc_poly_models[model_name] = svc_poly


for name, model in svc_poly_models.items():
    plot_decision_boundary(model, X_val_scaled, y_val, f'Decision Boundary: {name}')

gamma_values = [0.1, 10]
C_values = [0.01, 1, 100]

svc_rbf_models = {}

for gamma in gamma_values:
    for C_val in C_values:
        model_name = f'SVC_rbf_gamma_{gamma}_C_{C_val}'
        svc_rbf = SVC(kernel='rbf', gamma=gamma, C=C_val, random_state=42)
        svc_rbf.fit(X_train_scaled, y_train)
        svc_rbf_models[model_name] = svc_rbf


for name, model in svc_rbf_models.items():
    plot_decision_boundary(model, X_val_scaled, y_val, f'Decision Boundary: {name}')

param_grid_poly = {
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4],
    'coef0': [0, 1, 10]
}

grid_search_poly = GridSearchCV(
    SVC(kernel='poly', random_state=42),
    param_grid=param_grid_poly,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search_poly.fit(X_train_scaled, y_train)

print(f'Best parameters for SVC with polynomial kernel: {grid_search_poly.best_params_}')
print(f'Best accuracy: {grid_search_poly.best_score_:.4f}')
