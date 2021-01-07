#!/usr/bin/env python
# -*- coding=utf-8 -*-
from .MLargs import args
from .MLpipe import Pipe


def run():
    opt = args()
    wflow = Pipe(opt)
    wflow.pipe()
