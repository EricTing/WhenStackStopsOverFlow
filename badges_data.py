#!/usr/bin/env python

import luigi
import pandas as pd


class ExtractBadge(luigi.Task):
    starting_date = luigi.Parameter()

    def output(self):
        ofn = "earned_badges.{}.json".format(self.starting_date)
        return luigi.LocalTarget(ofn)

    def run(self):
        starting_date = self.starting_date
        qa = pd.read_json("./extracted.{}.json".format(starting_date))
        questioner = pd.read_json("./questioner.json")
        questioner.index = range(questioner.shape[0])

        qaq = pd.merge(qa, questioner, left_on='id', right_on='id')
        badges = pd.read_json("./badges.json")

        def foo(row):
            owneruserid = row['owneruserid']
            creationdate = row['creationdate']
            my_badges = ' '.join(badges[(badges['userid'] == owneruserid) & (
                badges['date'] < creationdate)]['name'])
            return my_badges

        earned = qaq.apply(foo, axis=1)
        print("earned shape: {}".format(earned.shape))
        print("{} out {} does not have badges".format(earned[
            earned == ''].shape[0], earned.shape[0]))
        qaq['badges'] = earned
        df = qaq[['badges', 'owneruserid', 'id']]
        df.columns = ['badges', 'userid', 'id']
        df.to_json(self.output().path)


def main(starting_date):
    qa = pd.read_json("./extracted.{}.json".format(starting_date))

    questioner = pd.read_json("./questioner.json")
    questioner.index = range(questioner.shape[0])

    qaq = pd.merge(qa, questioner, left_on='id', right_on='id')

    badges = pd.read_json("./badges.json")

    qaq_b = pd.merge(qaq,
                     badges[['date', 'name', 'userid']],
                     left_on='owneruserid',
                     right_on='userid')

    qaq_b = qaq_b[qaq_b['date'] < qaq_b['creationdate'].astype(
        'datetime64[ns]')]

    earned_badges = qaq_b.groupby(['userid', 'id']).apply(
        lambda g: ' '.join(g['name']))

    df = pd.DataFrame()
    df['userid'] = earned_badges.index.get_level_values(0)
    df['id'] = earned_badges.index.get_level_values(1)
    df['badges'] = earned_badges.values

    df.to_json("earned_badges.{}.json".format(starting_date),
               date_format='iso')


if __name__ == '__main__':
    luigi.build(
        [
            ExtractBadge(starting_date="2016-02-01"),
            ExtractBadge(starting_date="2015-11-01"),
        ],
        local_scheduler=True)
