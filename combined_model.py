#!/usr/bin/env python

import pprint
import luigi
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
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
                BADGES,
            ]))
        ]

        pipeline = Pipeline(feature_union + [
            ('dim', TruncatedSVD()),
            ('cls', LogisticRegression()),
        ])

        parameters = {
            "union__title__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__title__tfidf__min_df": [2, 4, 8],
            "union__paragraphs__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__paragraphs__tfidf__min_df": [2, 4, 8],
            "union__tags__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__tags__tfidf__min_df": [2, 4, 8],
            "union__badges__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__badges__tfidf__min_df": [2, 4, 8],
            "cls__C": [0.1, 1, 10],
            "dim__n_components": [100, 200, 300],
        }

        return pipeline, parameters

    def run(self):
        df = self.readData()

        pipeline, parameters = self.features()

        with open(self.output().path, 'w') as ofs:
            pprint.pprint(parameters, ofs)

            grid_search = GridSearchCV(pipeline,
                                       parameters,
                                       verbose=3,
                                       n_jobs=self.n_jobs,
                                       scoring='roc_auc',
                                       cv=3)
            grid_search.fit(df[feature_cols + ['badges']], df['success'])

            ofs.write("Best score: %0.3f\n" % grid_search.best_score_)
            ofs.write("Best parameters set:\n")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                ofs.write("\t%s: %r\n" %
                          (param_name, best_parameters[param_name]))

    def output(self):
        ofn = "/work/jaydy/WhenStackStopsOverFlow/combined_model.{}.txt".format(
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
        return df

    def features(self):
        feature_union = [
            ('union', FeatureUnion(transformer_list=[
                TITLE,
                PARAGRAPHS,
                TAGS,
                BADGES,
            ]))
        ]

        pipeline = Pipeline(feature_union + [
            ('dim', TruncatedSVD()),
            ('cls', LinearRegression()),
        ])

        parameters = {
            "union__title__tfidf__max_df": [0.4],
            "union__title__tfidf__min_df": [1],
            "union__paragraphs__tfidf__max_df": [0.4],
            "union__paragraphs__tfidf__min_df": [10],
            "union__tags__tfidf__max_df": [0.6],
            "union__tags__tfidf__min_df": [1],
            "dim__n_components": [20],
        }

        return pipeline, parameters

    def run(self):
        df = self.readData()

        pipeline, parameters = self.features()

        with open(self.output().path, 'w') as ofs:
            pprint.pprint(parameters, ofs)

            grid_search = GridSearchCV(pipeline,
                                       parameters,
                                       verbose=3,
                                       n_jobs=self.n_jobs,
                                       scoring='mean_absolute_error',
                                       cv=3)
            grid_search.fit(df[feature_cols + ['badges']], df['ElapsedTime'])

            ofs.write("Best score: %0.3f\n" % grid_search.best_score_)
            ofs.write("Best parameters set:\n")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                ofs.write("\t%s: %r\n" %
                          (param_name, best_parameters[param_name]))

    def output(self):
        ofn = "/work/jaydy/WhenStackStopsOverFlow/combined_model_time.{}.txt".format(
            self.starting_date)
        return luigi.LocalTarget(ofn)


def main():
    luigi.build(
        [
            CombinedModel(starting_date='2016-02-01',
                          n_jobs=2),
            # CombinedModelTime(starting_date='2016-02-01',
            #                   n_jobs=3),
        ],
        local_scheduler=True)


if __name__ == '__main__':
    main()
