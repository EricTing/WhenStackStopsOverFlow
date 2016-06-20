#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("darkgrid")
sns.set_context("poster", font_scale=1.3)

import pprint
import luigi
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.grid_search import GridSearchCV
from feature_union import ItemSelector, wordnet
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from build_model import feature_cols
from build_model import readData, readTimeDf
from badge_model import BadgeTimeDf

TITLE = ('title', Pipeline([
    ('selector', ItemSelector(key='title')),
    ('tfidf', TfidfVectorizer(tokenizer=wordnet,
                              stop_words='english')),
]))

PARAGRAPHS = ('paragraphs', Pipeline([
    ('selector', ItemSelector(key='paragraphs')), ('tfidf', TfidfVectorizer(
        tokenizer=wordnet, stop_words='english'))
]))

TAGS = ('tags', Pipeline([
    ('selector', ItemSelector(key='tags')), ('tfidf', TfidfVectorizer(
        token_pattern=r'(?u)\b\S+\b'))
]))

BADGES = ('badges', Pipeline([
    ('selector', ItemSelector(key='badges')), ('tfidf', TfidfVectorizer())
]))


class CombinedModel(luigi.Task):
    starting_date = luigi.Parameter()
    n_jobs = luigi.Parameter(default=8)

    def requires(self):
        pass

    def readData(self):
        starting_date = self.starting_date
        content_df = readData(starting_date=starting_date)
        badge_df = pd.read_json(BadgeTimeDf(starting_date=
                                            starting_date).output().path)
        df = pd.merge(content_df,
                      badge_df,
                      left_on='id',
                      right_on='id',
                      how='outer')
        df = df.drop_duplicates('id')
        df['badges'] = df['badges'].fillna(value='')
        return df

    def features(self):
        feature_union = [
            ('union', FeatureUnion(transformer_list=[
                TITLE,
                PARAGRAPHS,
                TAGS,
            ]))
        ]

        pipeline = Pipeline(feature_union + [
            ('dim', TruncatedSVD()),
            ('cls', LogisticRegression()),
        ])

        parameters = {
            "union__title__tfidf__max_df": [0.6],
            "union__title__tfidf__min_df": [2],
            "union__paragraphs__tfidf__max_df": [0.6],
            "union__paragraphs__tfidf__min_df": [8],
            "union__paragraphs__tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
            "union__tags__tfidf__max_df": [0.6],
            "union__tags__tfidf__min_df": [2],
            "cls__C": [0.1, 1, 10],
            "dim__n_components": [100],
        }

        return pipeline, parameters

    def run(self):
        df = self.readData()

        pipeline, parameters = self.features()

        pprint.pprint(parameters)

        grid_search = GridSearchCV(pipeline,
                                   parameters,
                                   verbose=3,
                                   n_jobs=self.n_jobs,
                                   scoring='roc_auc',
                                   cv=3)
        grid_search.fit(df[feature_cols + ['badges']], df['success'])

        pprint.pprint("Best score: %0.3f\n" % grid_search.best_score_)
        pprint.pprint("Best parameters set:\n")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            pprint.pprint("\t%s: %r\n" %
                          (param_name, best_parameters[param_name]))

        joblib.dump(grid_search.best_estimator_,
                    self.output().path,
                    compress=1)

    def output(self):
        ofn = "/work/jaydy/WhenStackStopsOverFlow/combined_model.{}.pkl".format(
            self.starting_date)
        return luigi.LocalTarget(ofn)


class CombinedModelTime(CombinedModel):
    def readData(self):
        starting_date = self.starting_date
        content_df = readTimeDf(starting_date=starting_date)
        badge_df = pd.read_json(BadgeTimeDf(starting_date=
                                            starting_date).output().path)
        df = pd.merge(content_df,
                      badge_df[['id', 'badges']],
                      left_on='id',
                      right_on='id',
                      how='outer')
        df = df.drop_duplicates('id')
        df['badges'] = df['badges'].fillna(value='')
        df = df[~df['ElapsedTime'].isnull()]
        df = df[df['ElapsedTime'] > 0]
        return df

    def features(self):
        feature_union = [
            ('union', FeatureUnion(transformer_list=[
                TITLE,
                PARAGRAPHS,
                TAGS,
            ]))
        ]

        pipeline = Pipeline(feature_union + [
            ('dim', TruncatedSVD()),
            ('cls', LinearRegression()),
        ])

        parameters = {
            "union__title__tfidf__max_df": [0.6],
            "union__title__tfidf__min_df": [2],
            "union__paragraphs__tfidf__max_df": [0.6],
            "union__paragraphs__tfidf__min_df": [8],
            "union__paragraphs__tfidf__ngram_range": [(1, 1)],
            "union__tags__tfidf__max_df": [0.6],
            "union__tags__tfidf__min_df": [2],
            "dim__n_components": [100],
        }

        return pipeline, parameters

    def nonlinarFeatures(self):
        feature_union = [
            ('union', FeatureUnion(transformer_list=[
                TITLE,
                PARAGRAPHS,
                TAGS,
            ]))
        ]

        pipeline = Pipeline(feature_union + [
            ('dim', TruncatedSVD()),
            ('cls', RandomForestRegressor(n_estimators=30)),
        ])

        parameters = {
            "union__title__tfidf__max_df": [0.6],
            "union__title__tfidf__min_df": [2],
            "union__paragraphs__tfidf__max_df": [0.6],
            "union__paragraphs__tfidf__min_df": [8],
            "union__paragraphs__tfidf__ngram_range": [(1, 1)],
            "union__tags__tfidf__max_df": [0.6],
            "union__tags__tfidf__min_df": [2],
            "dim__n_components": [100],
        }

        return pipeline, parameters

    def run(self):
        df = self.readData()

        pipeline, parameters = self.features()

        pprint.pprint(parameters)

        grid_search = GridSearchCV(pipeline,
                                   parameters,
                                   verbose=3,
                                   n_jobs=self.n_jobs,
                                   scoring='mean_absolute_error',
                                   cv=3)
        grid_search.fit(df[feature_cols + ['badges']],
                        np.log(df['ElapsedTime']))

        pprint.pprint("Best score: %0.3f\n" % grid_search.best_score_)
        pprint.pprint("Best parameters set:\n")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            pprint.pprint("\t%s: %r\n" %
                          (param_name, best_parameters[param_name]))

        joblib.dump(grid_search.best_estimator_,
                    self.output().path,
                    compress=1)

    def output(self):
        ofn = "/work/jaydy/WhenStackStopsOverFlow/combined_model_time.{}.pkl".format(
            self.starting_date)
        return luigi.LocalTarget(ofn)


def TimeModelEval():
    task = CombinedModelTime(starting_date='2016-02-01')
    df = task.readData()

    def linearFit2():
        pipeline, parameters = task.features()

        X = df[feature_cols + ['badges']]
        y = df['ElapsedTime']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.50)

        pipeline.fit(X_train, y_train)
        y_predict = pipeline.predict(X_test)

        error = metrics.mean_absolute_error(y_predict, y_test)

        plt.figure()
        plt.scatter(y_predict, y_test, alpha=0.3)
        plt.xlabel("Predicted response time [Min]")
        plt.ylabel("Actual response time [Min]")
        plt.title(
            "Predicted v.s. actual response time\nMean absolute error = {}".format(
                error))
        plt.xlim((0, max(y_predict)))
        plt.ylim((0, max(y_test)))
        plt.savefig("./linear_reg_scatter.png")

    def linearFit2Loga():
        pipeline, parameters = task.features()

        X = df[feature_cols + ['badges']]
        y = df['ElapsedTime']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.50)

        pipeline.fit(X_train, np.log10(y_train))
        y_predict = map(lambda x: 10**x, pipeline.predict(X_test))

        error = metrics.mean_absolute_error(y_predict, y_test)

        plt.figure()
        plt.scatter(y_predict, y_test, alpha=0.3)
        plt.xlabel("Predicted response time [Min]")
        plt.ylabel("Actual response time [Min]")
        plt.title(
            "Predicted v.s. actual response time\nMean absolute error = {}".format(
                error))
        plt.xlim((0, max(y_predict)))
        plt.ylim((0, max(y_test)))
        plt.savefig("./linear_log_reg_scatter.png")

    def randomForestFit():
        pipeline, _ = task.nonlinarFeatures()

        X = df[feature_cols + ['badges']]
        y = df['ElapsedTime']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.50)

        pipeline.fit(X_train, y_train)
        y_predict = pipeline.predict(X_test)

        error = metrics.mean_absolute_error(y_predict, y_test)

        plt.figure()
        plt.scatter(y_predict, y_test, alpha=0.3)
        plt.xlabel("Predicted response time [Min]")
        plt.ylabel("Actual response time [Min]")
        plt.title(
            "Predicted v.s. actual response time\nMean absolute error = {}".format(
                error))
        plt.xlim((0, max(y_predict)))
        plt.ylim((0, max(y_test)))
        plt.savefig("./randomforest_reg_scatter.png")

    linearFit2()
    linearFit2Loga()
    randomForestFit()


def main():
    luigi.build(
        [
            CombinedModel(starting_date='2016-02-01',
                          n_jobs=3),
        ],
        local_scheduler=True)
    TimeModelEval()


if __name__ == '__main__':
    main()
