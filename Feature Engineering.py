#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

from itertools import combinations
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

sns.set()
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("darkgrid")
sns.set_context("poster", font_scale=1.3)

import re
from bs4 import BeautifulSoup
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to sklearn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TitleParagrahsTagsExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        features = np.recarray(shape=(len(df), ),
                               dtype=[('title', object), (
                                   'paragraphs', object), ('tags', object)])

        for idx, row in df.iterrows():
            title = row['title']
            tags = re.sub(r"<|>", " ", row['tags'])

            body = row['body']
            soup = BeautifulSoup(body, 'lxml')
            paragraphs = soup.find_all('p')
            paragraphs = '\n'.join([_.getText() for _ in paragraphs])

            features['title'][idx] = title
            features['paragraphs'][idx] = paragraphs
            features['tags'][idx] = tags

        return features


class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


pipeline = Pipeline([
    # Extract title, paragraphs in the body, tags
    ("titleparagraphstags", TitleParagrahsTagsExtractor()),

    # Use FeatureUnion to combine the features from title, paragraphs and tags
    ('union',
     FeatureUnion(transformer_list=[
         ('title', Pipeline([
             ('selector', ItemSelector(key='title')),
             ('tfidf', TfidfVectorizer(min_df=100)),
         ])), ('paragraphs', Pipeline([
             ('selector', ItemSelector(key='paragraphs')),
             ('tfidf', TfidfVectorizer(min_df=100))
         ])), ('tags', Pipeline([
             ('selector', ItemSelector(key='tags')),
             ('tfidf', TfidfVectorizer(min_df=100))
         ]))
     ],

                  # weight components in FeatureUnion
                  transformer_weights={
                      'title': 0.1,
                      'paragraphs': 0.8,
                      'tags': 0.1,
                  }, )),
    ('to_dense', DenseTransformer()),
    # Use a naive bayes on the combined features
    ('nb', GaussianNB()),
])


def main():
    dbname = 'stackoverflow'
    username = 'jojo'
    pswd = 'iAmPass'

    con = None
    con = psycopg2.connect(database=dbname,
                           user=username,
                           host='localhost',
                           password=pswd)

    starting_date = '2016-03-05'
    print("Consider data after {}".format(starting_date))

    sql_query = """
    SELECT id, acceptedanswerid, creationdate, body, tags, title FROM posts
    where (posttypeid = 1) and creationdate > '{starting_date}'
    limit 4000
    ;
    """.format(starting_date=starting_date)
    questions = pd.read_sql_query(sql_query, con)

    df = questions[['id', 'body', 'tags', 'title']]

    df['FailedQuestion'] = np.random.randint(2, size=df.shape[0])
    df = df[['title', 'tags', 'body', 'FailedQuestion']]

    train = df.sample(2000)
    train.index = range(train.shape[0])
    test = df.sample(2000)
    test.index = range(test.shape[0])

    pipeline.fit(train[['title', 'body', 'tags']], train['FailedQuestion'])
    y = pipeline.predict(test[['title', 'body', 'tags']])
    print(classification_report(y, test['FailedQuestion']))


if __name__ == '__main__':
    main()
