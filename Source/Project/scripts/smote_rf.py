import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

# in order to check
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv('../data/processed_train_filled.csv')
df.replace(["unknown"], np.NaN, inplace=True)
non_null_df = df.dropna().copy()
non_null_df['default'] = non_null_df['default'].replace(['yes', 'no'], [1, 0])
non_null_df['housing'] = non_null_df['housing'].replace(['yes', 'no'], [1, 0])
non_null_df['loan'] = non_null_df['loan'].replace(['yes', 'no'], [1, 0])

month_dic = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
             'nov': 11, 'dec': 12}
week_dic = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}

non_null_df['month_ord'] = non_null_df['month'].map(month_dic)
non_null_df['week_ord'] = non_null_df['day_of_week'].map(week_dic)

dum_df = pd.get_dummies(non_null_df[['job', 'marital', 'education', 'contact', 'poutcome']], dtype=int)
clean_df = pd.concat(
    [non_null_df.drop(['job', 'marital', 'education', 'contact', 'poutcome', 'month', 'day_of_week'], axis=1), dum_df],
    axis=1)

X = clean_df.drop(['id', 'subscribe'], axis=1)
y = clean_df['subscribe']
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=66)

sm = SMOTE(random_state=66)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

rf_classifier = RandomForestClassifier(max_depth=33, min_samples_split=3, n_estimators=159)
rf_classifier.fit(X_train_smote, y_train_smote)

y_pred_rf = rf_classifier.predict(X_test)
y_proba_rf_smote = rf_classifier.predict_proba(X_test)[:, 1]
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title("Random Forest Model With SMOTE Confusion matrix")
plt.show()

metrics_score = {
    'ROC AUC': roc_auc_score,
    'Brier Score': brier_score_loss,
    'Log Loss': log_loss,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1 Score': f1_score,
    'Accuracy': accuracy_score
}

scores = {}
for metric_name, metric_func in metrics_score.items():
    if 'proba' in metric_name or metric_name in ['Brier Score', 'Log Loss', 'ROC AUC']:
        score = metric_func(y_test, y_proba_rf_smote)
    else:
        score = metric_func(y_test, y_pred_rf)
    scores[metric_name] = score

for metric_name, score in scores.items():
    print(f"{metric_name}: {score:.3f}")
