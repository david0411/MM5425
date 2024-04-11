import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# in order to check
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# data prepossessing
# import the data
df1 = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

# merge the training dataset and testing dataset
merged_df = pd.concat([df1, df2])
merged_df = merged_df.drop(['id'], axis=1)

# this is for the later reverse part
label_encoders = {}


def encode_labels(df, column):
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder


def fill_unknown(df, column):
    unique_values = df[column].astype(str).unique()

    if "unknown" in unique_values:
        df[column].replace("unknown", np.nan, inplace=True)

        if column in ['default', 'loan']:
            fill_df = df.copy()
            fill_df.drop(['housing', 'education', 'marital',
                          'poutcome', 'subscribe'],
                         axis=1, inplace=True)

            feature_columns = fill_df.columns.drop(column)
            for feature in feature_columns:
                if fill_df[feature].dtype == "object":
                    encode_labels(fill_df, feature)

            predict_fill(fill_df, feature_columns, column)
            df[column] = fill_df[column]
        else:
            most_common = df[column].value_counts(dropna=True).idxmax()
            df[column].fillna(most_common, inplace=True)


def predict_fill(df, feature_columns, target_column, model=RandomForestClassifier()):
    known_data = df[df[target_column].notna()]
    unknown_data = df[df[target_column].isna()]

    X_known = known_data[feature_columns]
    y_known = known_data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X_known, y_known, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nThe accuracy of the model for predicting {target_column} is: {accuracy}")

    X_unknown = unknown_data[feature_columns]
    predicted_values = model.predict(X_unknown)

    df.loc[df[target_column].isna(), target_column] = predicted_values


for column in merged_df.columns:
    if merged_df[column].dtype == "object" or len(merged_df[column].unique()) < 20:
        fill_unknown(merged_df, column)
        encode_labels(merged_df, column)


def reverse_encode(df, column, label_encoders):
    if column in df.columns:
        label_encoder = label_encoders[column]
        df[column] = label_encoder.inverse_transform(df[column])


for column in merged_df.columns:
    if column in label_encoders:
        reverse_encode(merged_df, column, label_encoders)

print(f"\n{merged_df.head(30)}")

for column in merged_df.columns:
    if merged_df[column].dtype == "object" or len(merged_df[column].unique()) < 20:
        print(f"Value counts for column {column}:")
        print(merged_df[column].value_counts())
        print("\n")

train_df = merged_df[merged_df['subscribe'].notna()]
test_df = merged_df[merged_df['subscribe'].isna()]
test_df = test_df.drop(columns=['subscribe'])

dfs = [train_df, test_df]
for df in dfs:
    df.index += 1
    df.index.name = "id"

train_df.to_csv("processed_train_filled.csv")
test_df.to_csv("processed_test_filled.csv")
