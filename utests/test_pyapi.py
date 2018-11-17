'''
Created on Nov 16, 2018

@author: fxua
'''
import numpy as np
from pyapi.forex_tick_predictor import ForexTickPredictor

predictor = ForexTickPredictor()

p = np.random.random(500)
predictor.loadAModel("eurusd_buy_1.flt")
predictor.loadAModel("eurusd_buy_2.flt")
predictor.setPeriods(150, 500)
predictor.loadTicks(p)
predictor.setFeatureNames("DMA,RSI,ROC,EMA,KAMA,LAG")

predictor.classifyATick(0.5)