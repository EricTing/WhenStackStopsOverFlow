#!/usr/bin/env python

import luigi
import pickle
import pprint
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion
from pymining import itemmining, assocrules, perftesting

from combined_model import CombinedModel
from combined_model import TITLE, PARAGRAPHS, TAGS, BADGES
from build_model import feature_cols


class BadgesModel(CombinedModel):
    def features(self):
        feature_union = [('union', FeatureUnion(transformer_list=[BADGES, ]))]

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


class TagsProduct(CombinedModel):
    def features(self):
        feature_union = [('union', FeatureUnion(transformer_list=[TAGS, ]))]

        pipeline = Pipeline(feature_union + [
            ('dim', TruncatedSVD()),
            ('cls', LogisticRegression()),
        ])

        parameters = {
            # "union__tags__tfidf__max_df": [0.6, 0.4, 0.2],
            # "union__tags__tfidf__min_df": [1, 4, 8],
            # "cls__C": [0.1, 1, 10],
            # "dim__n_components": [100, 200, 300],
            "union__tags__tfidf__max_df": [0.6],
            "union__tags__tfidf__min_df": [1],
            "cls__C": [0.1],
            "dim__n_components": [100],

            # [CV]  union__tags__tfidf__max_df=0.6, dim__n_components=100, union__tags__tfidf__min_df=1, cls__C=0.1, score=0.663602 -  12.6s
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
        ofn = "/work/jaydy/WhenStackStopsOverFlow/nb_tags.{}.product.pkl".format(
            self.starting_date)
        return luigi.LocalTarget(ofn)


class TagsProfiles(TagsProduct):
    def run(self):
        df = self.readData()
        transactions = df['tags'].apply(lambda s: s.split())
        relim_input = itemmining.get_relim_input(transactions)
        item_sets = itemmining.relim(relim_input, min_support=10)
        rules = assocrules.mine_assoc_rules(item_sets,
                                            min_support=10,
                                            min_confidence=0.5)
        dat = {t[0]: t[1:] for t in rules}
        pickle.dump(dat, open(self.output().path, 'wb'))

    def output(self):
        ofn = "./tags.{}.profile.pkl".format(self.starting_date)
        return luigi.LocalTarget(ofn)


def main():
    luigi.build(
        [
            TagsProduct(starting_date='2016-02-01',
                        n_jobs=3),
            TagsProfiles(starting_date='2016-02-01'),
        ],
        local_scheduler=True)


if __name__ == '__main__':
    main()
