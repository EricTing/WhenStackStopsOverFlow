#!/usr/bin/env python

import luigi

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion

from combined_model import CombinedModel
from combined_model import TITLE, PARAGRAPHS, TAGS, BADGES


class BadgesModel(CombinedModel):
    def features(self):
        feature_union = [
            ('union', FeatureUnion(transformer_list=[
                BADGES,
            ]))
        ]

        pipeline = Pipeline(feature_union + [
            ('dim', TruncatedSVD()),
            ('cls', LogisticRegression()),
        ])

        parameters = {
            "union__badges__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__badges__tfidf__min_df": [2, 4, 8],
            "cls__C": [0.1, 1, 10],
            "dim__n_components": [20, 50, 100, 200],
        }

        return pipeline, parameters

    def output(self):
        ofn = "/work/jaydy/WhenStackStopsOverFlow/nb_badges.{}.txt".format(
            self.starting_date)
        return luigi.LocalTarget(ofn)


class ContentModel(CombinedModel):
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
            "union__title__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__title__tfidf__min_df": [2, 4, 8],
            "union__paragraphs__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__paragraphs__tfidf__min_df": [2, 4, 8],
            "union__tags__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__tags__tfidf__min_df": [2, 4, 8],
            "cls__C": [0.1, 1, 10],
            "dim__n_components": [100, 200, 300],
        }

        return pipeline, parameters

    def output(self):
        ofn = "/work/jaydy/WhenStackStopsOverFlow/nb_content.{}.txt".format(
            self.starting_date)
        return luigi.LocalTarget(ofn)

def main():
    pass


if __name__ == '__main__':
    main()
