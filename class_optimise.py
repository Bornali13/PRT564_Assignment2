import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

# Load and preprocess dataset
df = pd.read_csv("Final_data_robust_scaled.csv")
df.columns = df.columns.str.strip()
df['Outbreak'] = (df['Total number of birds testing positive'] >= 2).astype(int)

features = ['AvgTemp', 'AvgPressure', 'AvgWindSpeed', 'Latitude', 'Longitude']
X = df[features]
y = df['Outbreak']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Feature Selection (Optional) ---
# selector = SelectKBest(score_func=f_classif, k='all')  # use this inside pipelines if desired

# --- Logistic Regression Optimization ---
logreg_pipe = Pipeline([
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
logreg_params = {
    'logreg__C': [0.01, 0.1, 1, 10],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__solver': ['liblinear']
}
logreg_cv = GridSearchCV(logreg_pipe, logreg_params, cv=StratifiedKFold(5), scoring='f1')
logreg_cv.fit(X_train_scaled, y_train)
logreg_best = logreg_cv.best_estimator_

# --- Naive Bayes (no hyperparams needed) ---
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# --- Random Forest Optimization ---
rf_pipe = Pipeline([
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
])
rf_params = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [5, 10, None],
    'rf__min_samples_split': [2, 5]
}
rf_cv = GridSearchCV(rf_pipe, rf_params, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
rf_cv.fit(X_train_scaled, y_train)
rf_best = rf_cv.best_estimator_

# --- Evaluation ---
def evaluate_model(name, model, X, y_true):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred))
    print(f"{name} ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")
    
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()
    
    return y_prob

# Evaluate all models
y_prob_logreg = evaluate_model("Logistic Regression", logreg_best, X_test_scaled, y_test)
y_prob_nb = evaluate_model("Naive Bayes", nb, X_test_scaled, y_test)
y_prob_rf = evaluate_model("Random Forest", rf_best, X_test_scaled, y_test)

# --- Plot ROC & PR curves ---
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

precision_logreg, recall_logreg, _ = precision_recall_curve(y_test, y_prob_logreg)
precision_nb, recall_nb, _ = precision_recall_curve(y_test, y_prob_nb)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)

# Plot ROC
plt.figure(figsize=(10, 5))
plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_logreg):.2f})')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {roc_auc_score(y_test, y_prob_nb):.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Optimised_ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

# Plot Precision-Recall
plt.figure(figsize=(10, 5))
plt.plot(recall_logreg, precision_logreg, label='Logistic Regression')
plt.plot(recall_nb, precision_nb, label='Naive Bayes')
plt.plot(recall_rf, precision_rf, label='Random Forest')
plt.xlabel("Optimised_Recall")
plt.ylabel("Optimised_Precision")
plt.title("Optimised_Precision-Recall Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()
