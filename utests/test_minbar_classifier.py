'''
Created on Dec 13, 2018

@author: fxua
'''
from pyapi.forex_minbar_predictor import ForexMinBarPredictor
import numpy as np

predictor = ForexMinBarPredictor()

nameStr="WILLR,RSI,DX,ADX,ROC,TRIX,CMO,ATR,TSF,CCI,NATR,\
 MFI,MINUS_DI,MINUS_DM,MOM,PLUS_DI,PLUS_DM,AD,OBV,MIDPRICE,\
 AROONOSC,WCLPRICE,MEDPRICE,KAMA,MA,TRIMA,WMA,TEMA,EMA,BBANDS"
 
nameStr='RSI,ADX,CMO,MOM,DX,AROONOSC,CCI,MFI,MINUS_DI,MINUS_DM,NATR,ROC,PLUS_DI,PLUS_DM,WILLR,TRIX,BINOM'

predictor.setFeatureNames(nameStr)
predictor.setLookback(1440)

predictor.loadHistoryBarFile("EURUSD_labeled.csv")
predictor.loadAModel("EURUSD_buy.pkl")

vol = np.random.uniform(low=10,high=100,size=(200,1))
data = np.random.uniform(low=1.2,high=1.4,size=(200,4))

v = np.hstack((data,vol.astype(int)))
predictor.loadHistoryMinBars(v, 200, 5)

# predictor.labelHistoryMinBars()
predictor.classifyMinBar(1.62, 1.63, 1.61, 1.62,80.)