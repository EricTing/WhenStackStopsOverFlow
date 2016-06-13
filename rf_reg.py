#!/usr/bin/env python
"""RandomForestRegressor
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from build_model import feature_cols
from build_model import readTimeDf
from feature_union import unionFeature


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
    cls_df = readTimeDf(starting_date=starting_date)
    print("data size: {}".format(cls_df.shape))

    feature_union = unionFeature()
    pipeline = Pipeline(feature_union + [('cls', RandomForestRegressor(
        n_estimators=20)), ])

    parameters = {
        "union__title__tfidf__max_df": [0.8, 0.6, 0.4],
        "union__title__tfidf__min_df": [1],
        # "union__title__tfidf__max_features": [1000, 4000],
        "union__paragraphs__tfidf__max_df": [0.8, 0.6, 0.4],
        "union__paragraphs__tfidf__min_df": [1],
        # "union__paragraphs__tfidf__max_features": [1000, 4000],
        "union__tags__tfidf__max_df": [1.0, 0.8, 0.6, 0.4],
        "union__tags__tfidf__min_df": [1],
        # "union__tags__tfidf__max_features": [4000, 7000, 10000, 20000],
        "cls__max_depth": [10, 20, 30, 40, 50, 60]
    }

    pprint.pprint(parameters)

    grid_search = GridSearchCV(pipeline,
                               parameters,
                               n_jobs=8,
                               scoring='mean_absolute_error',
                               cv=3)
    grid_search.fit(cls_df[feature_cols], cls_df['ElapsedTime'])

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':
    main(starting_date="2016-02-01")
