#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import logging
log_fn = "./so.log"
logging.basicConfig(filename=log_fn, level=logging.INFO)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("darkgrid")
sns.set_context("poster", font_scale=1.3)

import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

import psycopg2


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
        features = np.recarray(
            shape=(len(df), ),
            dtype=[('title', object), ('paragraphs', object), ('id', object), (
                'posttypeid', object), ('acceptedanswerid', object), (
                    'creationdate', object), ('tags', object), (
                        'parentid', object), ('codes', object)])

        idx = 0
        for _, row in df.iterrows():
            try:
                title = row['title']

                tags = None
                if row['tags'] is not None:
                    tags = re.sub(r"<|>", " ", row['tags'])

                body = row['body']
                soup = BeautifulSoup(body, 'lxml')
                paragraphs = soup.find_all('p')
                paragraphs = '\n'.join([_.getText() for _ in paragraphs])

                codes = None
                mycodes = soup.find_all('code')
                if len(mycodes) > 0:
                    codes = '\n'.join([_.getText() for _ in mycodes])

                myid = row['id']
                acceptedanswerid = row['acceptedanswerid']
                creationdate = row['creationdate']

                features['title'][idx] = title
                features['paragraphs'][idx] = paragraphs
                features['tags'][idx] = tags
                features['codes'][idx] = codes
                features['id'][idx] = myid
                features['acceptedanswerid'][idx] = acceptedanswerid
                features['creationdate'][idx] = creationdate
                features['posttypeid'][idx] = row['posttypeid']
                features['parentid'][idx] = row['parentid']

                idx += 1
            except Exception, e:
                logging.warning(e)

        return features


class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


def wordnet(text):
    tokens = word_tokenize(text)

    my_stemmer = WordNetLemmatizer()
    tokens = [my_stemmer.lemmatize(t) for t in tokens]

    return tokens


class SparseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return scipy.sparse.csr_matrix(X).T

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


def unionFeature(title_min_df=1,
                 title_max_df=0.9,
                 paragraphs_min_df=1,
                 paragraphs_max_df=0.9,
                 tags_min_df=1,
                 tags_max_df=0.9):
    """
    Keyword Arguments:
    title_min_df                        -- (default 1)
    title_max_df                        -- (default 0.9)

    paragraphs_min_df -- (default 1)
    paragraphs_max_df                   -- (default 0.9)

    tags_min_df       -- (default 1)
    tags_max_df                         -- (default 0.9)
    """
    feature_union = [
        # Use FeatureUnion to combine the features from title, paragraphs and tags
        ('union',
         FeatureUnion(transformer_list=[
             ('title', Pipeline([
                 ('selector', ItemSelector(key='title')),
                 ('tfidf', TfidfVectorizer(tokenizer=wordnet,
                                           stop_words='english')),
             ])),
             ('paragraphs', Pipeline([
                 ('selector', ItemSelector(key='paragraphs')), (
                     'tfidf', TfidfVectorizer(tokenizer=wordnet,
                                              stop_words='english'))
             ])),
             ('tags', Pipeline([
                 ('selector', ItemSelector(key='tags')), ('tfidf',
                                                          TfidfVectorizer())
             ]))
             # , ('codes', Pipeline([
             #     ('selector', ItemSelector(key='hasCodes')),
             #     ('to_sparse', SparseTransformer())
             # ]))
         ],

                      # weight components in FeatureUnion
                      transformer_weights={
                          'title': 0.33,
                          'paragraphs': 0.33,
                          'tags': 0.33,
                      }, )),
    ]
    return feature_union


def main(starting_date):
    dbname = 'stackoverflow'
    username = 'jojo'
    pswd = 'iAmPass'

    con = None
    con = psycopg2.connect(database=dbname,
                           user=username,
                           host='localhost',
                           password=pswd)

    print("Consider data after {}".format(starting_date))

    sql_query = """
    SELECT id, acceptedanswerid, parentid, creationdate, body, tags, title, posttypeid FROM posts
    where (posttypeid = 1 or posttypeid = 2) and creationdate > '{starting_date}'
    ;
    """.format(starting_date=starting_date)
    qa = pd.read_sql_query(sql_query, con)

    extractor = TitleParagrahsTagsExtractor()
    extracted = extractor.transform(qa)

    df = pd.DataFrame.from_records(extracted)
    df.to_pickle("extracted.{}.pkl".format(starting_date))


if __name__ == '__main__':
    import sys
    starting_date = sys.argv[1]
    main(starting_date)
