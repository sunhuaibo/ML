#!/usr/bin/env python
# -*- coding=utf-8 -*-
from sklearn import preprocessing

class Preprocess():
    def __init__(self, x, method):
        self.x = x
        self.method = method
    def preprocess(self):
        func = {
            "standard_scaler" : preprocessing.StandardScaler(),
            "min_max" : preprocessing.MinMaxScaler(),
            "max_abs" : preprocessing.MaxAbsScaler(),
            "ordinal_encoder" : preprocessing.OrdinalEncoder(),
            "onehot_encoder" : preprocessing.OneHotEncoder(handle_unknown='ignore')
        }
        return(func[self.method])
    def run(self):
        x_new = self.preprocess().fit_transform(self.x)
        return(x_new)

