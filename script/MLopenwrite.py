#!/usr/bin/env python
# -*- coding=utf-8 -*-
import numpy as np
import pandas as pd

class MLopen(object):
    def __init__(self, inpath):
        self.inpath = inpath
    def mlopen(self):
        df = pd.read_csv(self.inpath, sep="\t", header=None)
        df = np.array(df)
        return(df)

class MLwrite(object):
    def __init__(self, outfile, outpath):
        self.outfile = outfile
        self.outpath = outpath
    def mlwrite(self):
        self.outfile.to_csv(self.outpath, sep="\t", index=False)