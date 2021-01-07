#!/usr/bin/env python
# -*- coding=utf-8 -*-
from argparse import ArgumentParser
def args():
    parser = ArgumentParser()
    parser.add_argument("-x", help="Training data, shape (n_samples, n_features).", required=True)
    parser.add_argument("-y", help="Target values (classification: class labels , regression: real numbers). Two clumns, 1st is sampleID, 2st is label", required=True)
    parser.add_argument("-t", help="Type of feature")

    parse = parser.parse_args()
    return(parse)
