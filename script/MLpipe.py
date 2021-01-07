#!/usr/bin/env python
# -*- coding=utf-8 -*-
import numpy as np
import pandas as pd

from .MLopenwrite import MLopen, MLwrite
from .MLpreprocessing import Preprocess
from .MLestimator import Estimator

class Pipe(object):
    def __init__(self, args):
        self.args = args
    def pipe(self):
        x = MLopen(self.args.x).mlopen()
        y = MLopen(self.args.y).mlopen().ravel()

        preprocess = Preprocess(x, "standard_scaler")
        x_new = preprocess.run()

        estimator = Estimator(x_new, y, "svc")
        clf = estimator.run()
        print(clf.score(x_new, y))
