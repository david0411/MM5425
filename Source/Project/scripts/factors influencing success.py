import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def categorize_age(age):
    if age < 35:
        return 'age<35'
    elif 35 <= age <= 55:
        return 'age35-55'
    elif 56 <= age <= 80:
        return 'age56-80'
    else:
        return 'age>80'


df = pd.read_csv('../data/processed_train_filled.csv')
df['age_category'] = df['age'].apply(categorize_age)
df_with_dummies = pd.get_dummies(df, columns=['age_category'])
df_encoded_new = pd.get_dummies(df_with_dummies,
                                columns=['job', 'marital', 'education', 'contact', 'poutcome', 'default', 'housing',
                                         'loan'])
df_encoded_new['subscribe'] = df_encoded_new['subscribe'].map({'yes': 1, 'no': 0})
df_encoded_new['subscribe'].value_counts()
df_encoded_new['success'] = (df_encoded_new['subscribe'] == 1).astype(int)
df_encoded_new['success'].value_counts()

features = ['age_category_age<35', 'age_category_age35-55', 'age_category_age56-80', 'age_category_age>80',
            'housing_yes', 'housing_no', 'loan_yes', 'loan_no', 'job_admin.', 'job_blue-collar',
            'job_entrepreneur', 'job_housemaid', 'job_management',
            'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed',
            'marital_divorced', 'marital_married', 'marital_single',
            'education_basic.4y', 'education_basic.6y', 'education_basic.9y', 'education_high.school',
            'education_illiterate', 'education_professional.course', 'education_university.degree',
            'duration', 'campaign', 'previous']

target = ['success']

X = df_encoded_new[features]
y = df_encoded_new[target]
print(y.groupby('success').size())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)
# print(y_train.groupby('success').size())
# print(y_test.groupby('success').size())
# y_train = np.ravel(y_train.values)
# y_test = np.ravel(y_test.values)

# gb_clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.01, max_depth=7, min_samples_split=10)
# gb_clf.fit(X_train, y_train)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = np.ravel(y_train.values)
y_test = np.ravel(y_test.values)

model = GradientBoostingClassifier()

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_train, y_pred_train)
print("Training Confusion Matrix:")
print(cm)
cm = confusion_matrix(y_test, y_pred)
print("Testing Confusion Matrix:")
print(cm)

# y_pred_train_gb = gb_clf.predict(X_train)
# print(y_pred_train_gb.shape)
# print(accuracy_score(y_train, y_pred_train_gb, normalize=True))
# y_pred_gb = gb_clf.predict(X_test)
# print(y_pred_gb.shape)
# print(accuracy_score(y_test, y_pred_gb, normalize=True))
#
# feature_importance = gb_clf.feature_importances_
#
# # Sort feature importance scores and corresponding feature names
# sorted_indices = np.argsort(feature_importance)[::-1]
# sorted_features = np.array(features)[sorted_indices]
# sorted_importance = feature_importance[sorted_indices]
#
# conf_matrix_train = confusion_matrix(y_train, y_pred_train_gb)
# print("\nConfusion Matrix - Train Set Predictions:")
# print(conf_matrix_train)
# conf_matrix_test = confusion_matrix(y_test, y_pred_gb)
# print("Confusion Matrix - Test Set Predictions:")
# print(conf_matrix_test)

