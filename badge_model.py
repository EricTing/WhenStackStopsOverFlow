#!/usr/bin/env python

import pandas as pd
import numpy as np
import luigi
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from build_model import readResponseData


class BadgeSuccessDf(luigi.Task):
    starting_date = luigi.Parameter()

    def output(self):
        ofn = "/work/jaydy/WhenStackStopsOverFlow/badge_success_df.{}.json".format(
            self.starting_date)
        return luigi.LocalTarget(ofn)

    def run(self):
        starting_date = self.starting_date

        ifn = "earned_badges.{}.json".format(starting_date)

        badge_df = pd.read_json(ifn)

        q_a, questions, questions_unanswered_ids = readResponseData(
            starting_date=starting_date)

        shortest_elapsed_time = q_a.groupby('id_x').apply(
            lambda g: g['ElapsedTime'].min())

        print("One day = {} min".format(24 * 60))

        questions_answered_late_ids = shortest_elapsed_time[
            shortest_elapsed_time > 24 * 60].index

        failed_questions_ids = np.concatenate((questions_unanswered_ids.values,
                                               questions_answered_late_ids))

        questions['success'] = questions['id'].isin(
            failed_questions_ids).apply(lambda b: 0 if b else 1)

        badge_success_df = pd.merge(questions[['success', 'id']],
                                    badge_df[['badges', 'id']],
                                    left_on='id',
                                    right_on='id')

        badge_success_df.to_json(self.output().path)


class BadgeTimeDf(luigi.Task):
    starting_date = luigi.Parameter()

    def output(self):
        ofn = "/work/jaydy/WhenStackStopsOverFlow/badge_time_df.{}.json".format(
            self.starting_date)
        return luigi.LocalTarget(ofn)

    def run(self):
        starting_date = self.starting_date

        ifn = "earned_badges.{}.json".format(starting_date)

        badge_df = pd.read_json(ifn)

        q_a, _1, _2 = readResponseData(starting_date=starting_date)

        good_q_a = q_a[~q_a.acceptedanswerid_x.isnull()]

        df = good_q_a[['id_x', 'ElapsedTime']]
        df.columns = ['id', 'ElapsedTime']

        badge_time_df = pd.merge(df,
                                 badge_df[['badges', 'id']],
                                 left_on='id',
                                 right_on='id')
        badge_time_df.to_json(self.output().path)


class BadgeSuccessModelCV(luigi.Task):
    starting_date = luigi.Parameter()

    def requires(self):
        return BadgeSuccessDf(starting_date=self.starting_date)

    def run(self):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()), ('cls', LinearSVC())
        ])

        parameters = {
            "tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "tfidf__min_df": [1, 2, 3],
            "cls__C": [0.001, 0.01, 1.0, 10.0],
        }

        grid_search = GridSearchCV(pipeline,
                                   parameters,
                                   verbose=3,
                                   n_jobs=8,
                                   scoring='roc_auc',
                                   cv=3)

        df = pd.read_json(self.requires().output().path)
        grid_search.fit(df['badges'], df['success'])

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()), ('cls', RandomForestClassifier(
                n_estimators=20))
        ])

        parameters = {
            "tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "tfidf__min_df": [1, 2, 3],
            "cls__max_depth": [10, 20, 30, 40, 50, 60],
        }

        grid_search = GridSearchCV(pipeline,
                                   parameters,
                                   verbose=3,
                                   n_jobs=8,
                                   scoring='roc_auc',
                                   cv=3)

        df = pd.read_json(self.requires().output().path)
        grid_search.fit(df['badges'], df['success'])

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    def output(self):
        pass


class BadgeTimeModelCV(luigi.Task):
    starting_date = luigi.Parameter()

    def requires(self):
        return BadgeTimeDf(starting_date=self.starting_date)

    def run(self):
        ifn = self.requires().output().path
        df = pd.read_json(ifn)

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()), ('cls', RandomForestRegressor(
                n_estimators=20))
        ])

        parameters = {
            "tfidf__max_df": [1.0, 0.8, 0.6, 0.4, 0.2],
            "tfidf__min_df": [1, 2, 3],
            "cls__max_depth": [10, 20, 30, 40, 50, 60],
        }

        grid_search = GridSearchCV(pipeline,
                                   parameters,
                                   verbose=3,
                                   n_jobs=8,
                                   scoring='mean_absolute_error',
                                   cv=3)

        df = pd.read_json(self.requires().output().path)
        grid_search.fit(df['badges'], df['ElapsedTime'])

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    def output(self):
        pass


def main():
    luigi.build(
        [
            BadgeSuccessDf(starting_date="2016-02-01"),
            BadgeSuccessDf(starting_date="2015-11-01"),
            BadgeTimeDf(starting_date="2016-02-01"),
            BadgeTimeDf(starting_date="2015-11-01"),
            BadgeSuccessModelCV(starting_date="2016-02-01"),
            BadgeSuccessModelCV(starting_date="2015-11-01"),
            BadgeTimeModelCV(starting_date="2016-02-01"),
            BadgeTimeModelCV(starting_date="2015-11-01"),
        ],
        local_scheduler=True)


if __name__ == '__main__':
    main()
