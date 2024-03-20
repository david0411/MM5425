import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import graphviz

data_df = pd.read_csv('../data/train.csv')

data_dum_df = pd.concat([
    data_df,
    pd.get_dummies(data_df['job'].replace('unknown', 'unknown_job')),
    pd.get_dummies(data_df['education'].replace('unknown', 'unknown_edu')),
    pd.get_dummies(data_df['marital'].replace('unknown', 'unknown_mar')),
    pd.get_dummies(data_df['contact']),
    pd.get_dummies(data_df['month']),
    pd.get_dummies(data_df['day_of_week']),
    pd.get_dummies(data_df['poutcome'])
], axis=1)
features = ['duration', 'campaign', 'pdays', 'previous',
            'emp_var_rate', 'cons_price_index', 'cons_conf_index', 'lending_rate3m', 'nr_employed',
            'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services',
            'student', 'technician', 'unemployed', 'unknown_job',
            'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree',
            'unknown_edu',
            'divorced', 'married', 'single', 'unknown_mar',
            'cellular', 'telephone',
            'apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep',
            'fri', 'mon', 'thu', 'tue', 'wed']
target = ['subscribe']
X = data_dum_df[features]
Y = data_dum_df[target]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=99)

decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=250)
decision_tree.fit(X_train, Y_train)
treeObj = decision_tree.tree_
print('Node Count: ', treeObj.node_count, 'Depth: ', treeObj.max_depth)


Y_pred = decision_tree.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
target_names = data_dum_df['subscribe'].unique().tolist()
dot_data = tree.export_graphviz(decision_tree, feature_names=features, class_names=target_names, filled=True,
                                rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("full_tree")

print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

score_cv = cross_val_score(decision_tree, X, Y, cv=5)
print(score_cv)
print(score_cv.mean())
