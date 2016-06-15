#!/usr/bin/env python

import luigi
from nb_tags import BadgesModel

def main():
    luigi.build([
        BadgesModel(starting_date='2016-02-01', n_jobs=8)
    ],
                local_scheduler=True)

if __name__ == '__main__':
    main()
