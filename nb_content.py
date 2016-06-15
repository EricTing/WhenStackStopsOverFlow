#!/usr/bin/env python

from nb_tags import ContentModel
import luigi

def main():
    luigi.build([
        ContentModel(starting_date='2016-02-01', n_jobs=8)
    ],
                local_scheduler=True)

if __name__ == '__main__':
    main()
