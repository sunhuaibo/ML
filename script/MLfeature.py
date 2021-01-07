#!/usr/bin/env python
# -*- coding=utf-8 -*-

from sklearn.feature_selection import (VarianceThreshold, SelectKBest,chi2, 
                                       RFE, SelectFromModel, 
                                       SequentialFeatureSelector)

from .MLestimator import Estimator

class Feature(object):
    def __init__(self, args):
        self.args = args

    def feature(self):
        func = {
            "variance_threshold" : VarianceThreshold(),
            "select_Kbest" : SelectKBest(),
            "chi2" : chi2(),
            "rfe" : RFE(),
            "select_from_model" : SelectFromModel(),
            "sequential_feature_selector" : SequentialFeatureSelector(),
        }
        return(func[self.args.feature_method])
    def run(self):
        if self.method in (["rfe", "select_from_model", 
                            "sequential_feature_selector"]):
            estimator = Estimator(self.args.feature_estimator_method)
            