import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt

# --- Load Dataset ---
df = pd.read_csv("Final_data_robust_scaled.csv")  # Adjust path if needed
df.columns = df.columns.str.strip()

# --- Create Binary Target Variable ---
df['Outbreak'] = (df['Total number of birds testing positive'] >= 2).astype(int)

# --- Define Features and Target ---
features = ['AvgTemp', 'AvgPressure', 'AvgWindSpeed', 'Latitude', 'Longitude']
X = df[features]
y = df['Outbreak']

# --- Train-Test Split and Scaling ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Logistic Regression ---
logreg = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
param_logreg = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']}
grid_logreg = GridSearchCV(logreg, param_logreg, cv=StratifiedKFold(n_splits=5), scoring='f1')
grid_logreg.fit(X_train_scaled, y_train)
logreg_best = grid_logreg.best_estimator_

# --- Naive Bayes ---
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# --- Random Forest Classifier ---
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
param_rf = {
    'n_estimators': [100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
}
grid_rf = GridSearchCV(rf, param_rf, cv=StratifiedKFold(n_splits=5), scoring='f1')
grid_rf.fit(X_train_scaled, y_train)
rf_best = grid_rf.best_estimator_

# --- Evaluation Helper Function ---
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{name} ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    
    # Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()
    
    return y_prob

# --- Evaluate All Models ---
y_prob_logreg = evaluate_model("Logistic Regression", logreg_best, X_test_scaled, y_test)
y_prob_nb = evaluate_model("Naive Bayes", nb, X_test_scaled, y_test)
y_prob_rf = evaluate_model("Random Forest", rf_best, X_test_scaled, y_test)

# --- ROC and Precision-Recall Curves ---
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

precision_logreg, recall_logreg, _ = precision_recall_curve(y_test, y_prob_logreg)
precision_nb, recall_nb, _ = precision_recall_curve(y_test, y_prob_nb)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)

# --- Plot ROC Curves ---
plt.figure(figsize=(10, 5))
plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_logreg):.2f})')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {roc_auc_score(y_test, y_prob_nb):.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot Precision-Recall Curves ---
plt.figure(figsize=(10, 5))
plt.plot(recall_logreg, precision_logreg, label='Logistic Regression')
plt.plot(recall_nb, precision_nb, label='Naive Bayes')
plt.plot(recall_rf, precision_rf, label='Random Forest')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()
