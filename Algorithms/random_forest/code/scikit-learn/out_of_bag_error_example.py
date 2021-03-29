from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 123

# Generate a binary classification dataset.
X, y = make_classification(n_samples=500, n_features=25,
                           n_clusters_per_class=1, n_informative=15,
                           random_state=RANDOM_STATE)

model = RandomForestClassifier(oob_score=True, random_state=RANDOM_STATE)

model.fit(X, y)

print('Out of bag error:', model.oob_score_)
