#!/usr/bin/env python
"""Random Forest
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("darkgrid")
sns.set_context("poster", font_scale=1.3)

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.metrics import classification_report

from build_model import readData, feature_cols
from feature_union import unionFeature


def main(starting_date="2016-03-01"):
    cls_train, cls_test = readData(starting_date=starting_date)

    feature_union = unionFeature(title_min_df=1,
                                 title_max_df=0.7,
                                 paragraphs_min_df=10,
                                 paragraphs_max_df=0.7,
                                 tags_min_df=1,
                                 tags_max_df=0.7)

    pipeline = Pipeline(feature_union + [('cls', RandomForestClassifier(
        class_weight='balanced',
        n_estimators=50, n_jobs=16)), ])

    pipeline.fit(cls_train[feature_cols], cls_train['success'])
    y = pipeline.predict(cls_test[feature_cols])
    print(classification_report(y, cls_test['success']))

    joblib.dump(pipeline, "./randomforest.{}.pkl".format(starting_date))

    scores = pipeline.predict_proba(cls_test[feature_cols])[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(
        cls_test['success'], scores, pos_label=1)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig("./randomforest_roc.png")
    plt.title("ROC of randomforest classifier")
    print("Area under curve for randomforest is {}".format(metrics.auc(fpr,
                                                                       tpr)))


if __name__ == '__main__':
    main(starting_date="2016-03-01")
