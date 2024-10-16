import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize, scale
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                             f1_score, precision_recall_curve, roc_curve, 
                             auc, classification_report, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.utils import resample

import hashlib


def plot_decision_boundary(model, X, y, title):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
    plt.title(title)
    plt.show()

def evaluate_overfitting(model, X_train, y_train, X_val, y_val, model_name):

    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f'{model_name} - Training Accuracy: {train_score:.4f}, Validation Accuracy: {val_score:.4f}')
    if train_score > val_score + 0.05:
        print('-> The model may be overfitting.')
    else:
        print('-> The model does not show signs of overfitting.')
    print('-'*50)

def compute_metrics(y_true, y_pred, model_name):

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f'Metrics for {model_name}:')
    print('Confusion Matrix:')
    print(cm)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('-'*50)

def plot_pr_roc_curves(model, X_val, y_val, n_classes, model_name):

    y_val_binarized = label_binarize(y_val, classes=np.unique(y_val))
    
    if hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_val)
    else:
        y_scores = model.predict_proba(X_val)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_val_binarized[:, i], y_scores[:, i])
        average_precision[i] = auc(recall[i], precision[i])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_binarized[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=f'Class {i} (AUC = {average_precision[i]:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend()
    plt.show()
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend()
    plt.show()

#Start
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

plt.scatter(X_stretched[:, 0], X_stretched[:, 1], c=y_true, cmap='viridis', edgecolor='k')
plt.title('Dataset (a)')
plt.show()

X_train, X_val, y_train, y_val = train_test_split(
    X_stretched, y_true, test_size=0.2, random_state=42, stratify=y_true)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

linear_models = {
    'LinearSVC (C=1e5)': LinearSVC(C=1e5, max_iter=10000, random_state=42),
    'SVC (linear kernel C=1e5)': SVC(kernel='linear', C=1e5, random_state=42),
    'LinearSVC (C=100)': LinearSVC(C=100, max_iter=10000, random_state=42),
    'SVC (linear kernel C=100)': SVC(kernel='linear', C=100, random_state=42)
}

#degree = 1, 3, 5
#coef0 = 1, 10
svc_poly_models = {
    'SVC (poly degree=1 coef0=1)': SVC(kernel='poly', degree=1, coef0=1, C=1.0, random_state=42),
    'SVC (poly degree=3 coef0=1)': SVC(kernel='poly', degree=3, coef0=1, C=1.0, random_state=42),
    'SVC (poly degree=3 coef0=10)': SVC(kernel='poly', degree=3, coef0=10, C=1.0, random_state=42),
}

gamma_values = [0.1, 10]
C_values = [0.01, 1, 100]

svc_rbf_models = {}
for gamma in gamma_values:
    for C_val in C_values:
        model_name = f'SVC_rbf_gamma_{gamma}_C_{C_val}'
        svc_rbf = SVC(kernel='rbf', gamma=gamma, C=C_val, random_state=42)
        svc_rbf.fit(X_train_scaled, y_train)
        svc_rbf_models[model_name] = svc_rbf


for name, model in linear_models.items():
    model.fit(X_train_scaled, y_train)
    plot_decision_boundary(model, X_val_scaled, y_val, f'Decision Boundary: {name}')

for name, model in svc_poly_models.items():
    model.fit(X_train_scaled, y_train)
    plot_decision_boundary(model, X_val_scaled, y_val, f'Decision Boundary: {name}')

for name, model in svc_rbf_models.items():
    plot_decision_boundary(model, X_val_scaled, y_val, f'Decision Boundary: {name}')

for name, model in linear_models.items():
    evaluate_overfitting(model, X_train_scaled, y_train, X_val_scaled, y_val, name)
    y_pred = model.predict(X_val_scaled)
    compute_metrics(y_val, y_pred, name)

for name, model in svc_poly_models.items():
    evaluate_overfitting(model, X_train_scaled, y_train, X_val_scaled, y_val, name)
    y_pred = model.predict(X_val_scaled)
    compute_metrics(y_val, y_pred, name)

for name, model in svc_rbf_models.items():
    evaluate_overfitting(model, X_train_scaled, y_train, X_val_scaled, y_val, name)
    y_pred = model.predict(X_val_scaled)
    compute_metrics(y_val, y_pred, name)

n_classes = len(np.unique(y_true))

for name, model in linear_models.items():
    plot_pr_roc_curves(model, X_val_scaled, y_val, n_classes, name)

for name, model in svc_poly_models.items():
    plot_pr_roc_curves(model, X_val_scaled, y_val, n_classes, name)

for name, model in svc_rbf_models.items():
    plot_pr_roc_curves(model, X_val_scaled, y_val, n_classes, name)

param_grid_rbf = {
        "C": [0.1, 0.5, 1, 10, 100, 1000],
        "gamma": ["scale", 1, 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["rbf"],
}

grid_search_rbf = GridSearchCV(
    SVC(kernel='rbf', random_state=42),
    param_grid=param_grid_rbf,
    cv=10,
    scoring='accuracy'
)

grid_search_rbf.fit(X_train_scaled, y_train)

print(f'Best parameters for SVC with rbf kernel: {grid_search_rbf.best_params_}')
print(f'Best cross-validation accuracy: {grid_search_rbf.best_score_:.4f}')

best_svc_rbf = grid_search_rbf.best_estimator_

plot_decision_boundary(best_svc_rbf, X_val_scaled, y_val, 'Best SVC (rbf) Decision Boundary')

y_pred_best_rbf = best_svc_rbf.predict(X_val_scaled)
compute_metrics(y_val, y_pred_best_rbf, 'Best SVC (rbf)')
evaluate_overfitting(best_svc_rbf, X_train_scaled, y_train, X_val_scaled, y_val, 'Best SVC (rbf)')

plot_pr_roc_curves(best_svc_rbf, X_val_scaled, y_val, n_classes, 'Best SVC (rbf)')