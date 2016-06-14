#!/usr/bin/env python

import pandas as pd

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

    qaq_b = qaq_b[qaq_b['date'] < qaq_b['creationdate'].astype('datetime64[ns]')]

    earned_badges = qaq_b.groupby(['userid', 'id']).apply(
        lambda g: ' '.join(g['name']))

    df = pd.DataFrame()
    df['userid'] = earned_badges.index.get_level_values(0)
    df['id'] = earned_badges.index.get_level_values(1)
    df['badges'] = earned_badges.values

    df.to_json("earned_badges.{}.json".format(starting_date),
            date_format='iso')


if __name__ == '__main__':
    starting_date = "2016-02-01"
    main(starting_date)
    starting_date = "2015-11-01"
    main(starting_date)
