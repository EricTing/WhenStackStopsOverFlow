#!/usr/bin/env python

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import os
import xml.etree.cElementTree as etree
import logging

log_fn = "./dump_so.log"
logging.basicConfig(filename=log_fn, level=logging.INFO)

dbname = "stackoverflow"
username = "jojo"
pswd = 'iAmPass'

engine = create_engine('postgresql://%s:%s@localhost/%s' %
                       (username, pswd, dbname))
if not database_exists(engine.url):
    create_database(engine.url)

if database_exists(engine.url):
    print("successfully connect to {}".format(engine.url))
