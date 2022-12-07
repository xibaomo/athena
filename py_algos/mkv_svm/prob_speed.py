#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 03:53:55 2022

@author: naopc
"""
import pdb

class ProbSpeedConfig(object):
    def __init__(self, gencfg):
        self.yamlDict = gencfg.yamlDict
        self.root = "PROB_SPEED"

    def getProbNodes(self):
        return self.yamlDict[self.root]['PROB_NODES']

    def getMinSpeed(self):
        return self.yamlDict[self.root]['MIN_SPEED']
    def getZeroAccLimit(self):
        return self.yamlDict[self.root]['ZERO_ACC']

class ProbSpeedPredictor(object):
    def __init__(self, gencfg):
        self.cfg = ProbSpeedConfig(gencfg)

    def predict(self, fm):
        prob_buy = fm[0, 0]
        spd = fm[0, 1]

        if abs(spd) < self.cfg.getMinSpeed():
            print("Speed too low. No action. ",spd, self.cfg.getMinSpeed())
            return 0

        # if fm[0, 4] > self.cfg.getMaxAcc(): #acceleration
        #     return 0
        acc = fm[0, 4]
        if abs(acc) < self.cfg.getZeroAccLimit():
            acc = 0
        if fm[0, 1] * acc > 0:
            print("Speed and acceleration has the sanme sign. No action")
            return 0

        act = 0

        prob_low = self.cfg.getProbNodes()[0]
        prob_high = self.cfg.getProbNodes()[1]

        if prob_buy >= prob_high:
            act = 2
        if prob_buy <= prob_low:
            act = 1

        return act
