'''
Created on Nov 16, 2018

@author: fxua
'''
from pyapi.forex_tick_predictor import ForexTickPredictor

predictor = ForexTickPredictor()

p = [1,2,3,4,5.]

predictor.loadTicks(p)

predictor.showFeatureCalculator()