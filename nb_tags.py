#!/usr/bin/env python

import luigi

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import GaussianNB

from combined_model import CombinedModel
from combined_model import TITLE, PARAGRAPHS, TAGS, BADGES


class TagsModel(CombinedModel):
    def features(self):
        feature_union = [
            ('union', FeatureUnion(transformer_list=[
                TAGS,
            ]))
        ]

        pipeline = Pipeline(feature_union + [
            ('dim', TruncatedSVD()),
            ('norm', Normalizer()),
            ('cls', GaussianNB()),
        ])

        parameters = {
            "union__tags__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__tags__tfidf__min_df": [2, 4, 8],
            "dim__n_components": [100, 200, 300],
        }

        return pipeline, parameters

    def output(self):
        ofn = "/work/jaydy/WhenStackStopsOverFlow/nb_tags.{}.txt".format(
            self.starting_date)
        return luigi.LocalTarget(ofn)


class BadgesModel(CombinedModel):
    def features(self):
        feature_union = [
            ('union', FeatureUnion(transformer_list=[
                BADGES,
            ]))
        ]

        pipeline = Pipeline(feature_union + [
            ('dim', TruncatedSVD()),
            ('norm', Normalizer()),
            ('cls', GaussianNB()),
        ])

        parameters = {
            "union__badges__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__badges__tfidf__min_df": [2, 4, 8],
            "dim__n_components": [100, 200, 300],
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
            ]))
        ]

        pipeline = Pipeline(feature_union + [
            ('dim', TruncatedSVD()),
            ('norm', Normalizer()),
            ('cls', GaussianNB()),
        ])

        parameters = {
            "union__title__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__title__tfidf__min_df": [2, 4, 8],
            "union__paragraphs__tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "union__paragraphs__tfidf__min_df": [2, 4, 8],
            "dim__n_components": [100, 200, 300],
        }

        return pipeline, parameters

    def output(self):
        ofn = "/work/jaydy/WhenStackStopsOverFlow/nb_content.{}.txt".format(
            self.starting_date)
        return luigi.LocalTarget(ofn)

def main():
    luigi.build([
        TagsModel(starting_date='2016-02-01', n_jobs=8)
    ],
                local_scheduler=True)


if __name__ == '__main__':
    main()
