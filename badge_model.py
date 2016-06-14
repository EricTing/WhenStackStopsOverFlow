#!/usr/bin/env python

import pandas as pd
import numpy as np
import luigi
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


def main():
    luigi.build(
        [
            BadgeSuccessDf(starting_date="2016-02-01"),
            BadgeSuccessDf(starting_date="2015-11-01"),
            BadgeTimeDf(starting_date="2016-02-01"),
            BadgeTimeDf(starting_date="2015-11-01"),
        ],
        local_scheduler=True)


if __name__ == '__main__':
    main()
