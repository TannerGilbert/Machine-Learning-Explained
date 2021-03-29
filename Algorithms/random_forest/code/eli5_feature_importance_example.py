from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from IPython.display import display

import eli5
from eli5.sklearn import PermutationImportance

RANDOM_STATE = 0

# Get Iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Create and train Random Forest
model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)

display(eli5.show_weights(perm, feature_names=iris.feature_names))

eli5_weights = eli5.explain_weights(model, feature_names=iris.feature_names)
print(eli5_weights)