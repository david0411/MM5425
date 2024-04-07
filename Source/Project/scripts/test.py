import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
# combine test dataset and train dataset
df = pd.concat([train, test], axis=0)

df_no_unknown = df[(df['job'] != 'unknown') & (df['education'] != 'unknown') & (df['marital'] != 'unknown') & (
        df['default'] != 'unknown') & (df['housing'] != 'unknown') & (df['loan'] != 'unknown')]
features_list = ['job', 'education', 'marital', 'default', 'housing', 'loan']

one_hot_df = df_no_unknown.reset_index()
for feature in features_list:
    encoder = OneHotEncoder(sparse_output=False)
    feature_df = pd.DataFrame(encoder.fit_transform(df_no_unknown[[feature]]),
                              columns=encoder.get_feature_names_out([feature]))
    one_hot_df = pd.concat([one_hot_df, feature_df], axis=1)

print(one_hot_df)
