#!/usr/bin/env python
# -*- coding=utf-8 -*-
from sklearn import svm
from sklearn import ensemble

class Estimator():
    def __init__(self, x, y, method):
        self.x = x
        self.y = y
        self.method = method
    def estimator(self):
        func = {
            "svc" : svm.SVC(),
            "rf" : ensemble.RandomForestClassifier()
            }
        return(func[self.method])
    def run(self):
        clf = self.estimator()
        clf.fit(self.x, self.y)
        return(clf)

