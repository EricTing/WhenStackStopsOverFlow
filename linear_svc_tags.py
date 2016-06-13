#!/usr/bin/env python
"""lnear SVC
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("darkgrid")
sns.set_context("poster", font_scale=1.3)
import numpy as np
import pprint

from operator import itemgetter
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from build_model import readData, feature_cols
from feature_union import unionTagsFeature


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score, np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def main(starting_date="2016-03-01"):
    cls_df = readData(starting_date=starting_date)
    print("data size: {}".format(cls_df.shape))

    feature_union = unionTagsFeature()
    pipeline = Pipeline(feature_union + [('cls', LinearSVC(class_weight=
                                                           'balanced')), ])

    parameters = {
        "union__tags__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
        "union__tags__tfidf__min_df": [1],
        "cls__C": [0.001, 0.01, 1.0, 10.0]
    }

    pprint.pprint(parameters)

    grid_search = GridSearchCV(pipeline,
                               parameters,
                               verbose=3,
                               n_jobs=8,
                               scoring='roc_auc',
                               cv=3)
    grid_search.fit(cls_df[feature_cols], cls_df['success'])

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':
    main(starting_date="2016-02-01")
