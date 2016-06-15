#!/usr/bin/env python

import luigi
import pprint
import pandas as pd
from build_model import readData
from badge_model import BadgeTimeDf
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.grid_search import GridSearchCV
from feature_union import ItemSelector, wordnet
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from build_model import feature_cols


class CombinedModel(luigi.Task):
    starting_date = luigi.Parameter()

    def requires(self):
        pass

    def readData(self):
        starting_date = self.starting_date
        content_df = readData(starting_date=starting_date)
        badge_df = pd.read_json(BadgeTimeDf(starting_date=
                                            starting_date).output().path)
        df = pd.merge(content_df, badge_df, left_on='id', right_on='id')
        df = df.drop_duplicates('id')
        return df

    def run(self):
        df = self.readData()

        feature_union = [
            ('union', FeatureUnion(transformer_list=[
                ('title', Pipeline([
                    ('selector', ItemSelector(key='title')),
                    ('tfidf', TfidfVectorizer(tokenizer=wordnet,
                                              stop_words='english')),
                ])), ('paragraphs', Pipeline([
                    ('selector', ItemSelector(key='paragraphs')), (
                        'tfidf', TfidfVectorizer(tokenizer=wordnet,
                                                 stop_words='english'))
                ])), ('tags', Pipeline([
                    ('selector', ItemSelector(key='tags')), (
                        'tfidf', TfidfVectorizer(token_pattern=r'(?u)\b\S+\b'))
                ])), ('badges', Pipeline([
                    ('selector', ItemSelector(key='badges')), (
                        'tfidf', TfidfVectorizer())
                ]))
            ]))
        ]

        pipeline = Pipeline(feature_union + [
            ('dim', TruncatedSVD()), ('cls', LinearSVC())
        ])

        parameters = {
            "union__title__tfidf__max_df": [0.4],
            "union__title__tfidf__min_df": [1],
            "union__paragraphs__tfidf__max_df": [0.4],
            "union__paragraphs__tfidf__min_df": [10],
            "union__tags__tfidf__max_df": [0.6],
            "union__tags__tfidf__min_df": [1],
            "dim__n_components": [100],
            "cls__C": [0.01]
        }

        pprint.pprint(parameters)

        grid_search = GridSearchCV(pipeline,
                                   parameters,
                                   verbose=3,
                                   n_jobs=8,
                                   scoring='roc_auc',
                                   cv=3)
        grid_search.fit(df[feature_cols + ['badges']], df['success'])

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    def output(self):
        ofn = "/work/jaydy/WhenStackStopsOverFlow/combined_model.{}.txt".format(
            self.starting_date)
        return luigi.LocalTarget(ofn)


def main():
    luigi.build([CombinedModel(starting_date='2016-02-01')],
                local_scheduler=True)


if __name__ == '__main__':
    main()
