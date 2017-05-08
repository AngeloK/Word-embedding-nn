# coding=utf-8
#!/usr/bin/env python

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from read_data import read_reviews, read_vocabulary
from learning_curve import plot_learning_curve
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (train_test_split, cross_val_score,
                                    cross_val_predict, GridSearchCV)
from sklearn import svm
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt

categories = ['neg', 'pos']
base_path = '/Users/Neyanbhbin/Documents/code/cs585/project-angelok/aclImdb'
vocabulary_path = base_path + '/imdb.vocab'

vocabulary = read_vocabulary(vocabulary_path)

data, labels = read_reviews(base_path, data_type="train", labels=categories)
print("Data Shape")
print("(%d, %d)" %(len(data), len(vocabulary)))


# Encoding labels to numeric values.
le = LabelEncoder()
le.fit(categories)
y = le.transform(labels)


params = {
    "vocabulary": vocabulary,
    # "min_df": 3,
    # "use_idf": 1,
    # "stop_words": 'english'
}

estimators = [('tfidf_vec', TfidfVectorizer(**params)),
            #   ('clf', SGDClassifier())
              ('clf', LogisticRegression())
            ]
pipe = Pipeline(estimators)
# print(pipe.steps)
# params = dict(clf__alpha=[0.0001, 0.0003, 0.0006])
# grid_search = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1)
# grid_search.fit(data, y)
# print(grid_search.grid_scores_)

pipe.fit(data, y)

print(pipe.score(data, y))


# print(cross_val_score(pipe, data, y, cv=5))


def model_eva(model, params=None, dense_matrx=False):
    
    if params:
        m = model(*params)
    else:
        m = model()
    if dense_matrx:
        m.fit(X.toarray(), y_train)
        y_pred = m.predict(X_t.toarray())
    else:
        m.fit(X, y_train)
        y_pred = m.predict(X_t)
    return classification_report(y_test, y_pred, target_names=categories)


# title = 'Logistic Regression on IMDB Reviews Sentiment Analysis'
# plot_learning_curve(LogisticRegression(), title=title, X=X, y=y, cv=5, n_jobs=4)
# plt.show()

test_data, test_labels = read_reviews(base_path, data_type="test", labels=categories)
test_y = le.transform(test_labels)

print(pipe.score(test_data, test_y))
