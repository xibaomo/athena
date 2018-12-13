'''
Created on Dec 13, 2018

@author: fxua
'''
from pyapi.forex_minbar_predictor import ForexMinBarPredictor

predictor = ForexMinBarPredictor()

nameStr="WILLR,RSI,DX,ADX,ROC,TRIX,CMO,ATR,TSF,CCI,NATR,\
 MFI,MINUS_DI,MINUS_DM,MOM,PLUS_DI,PLUS_DM,AD,OBV,MIDPRICE,\
 AROONOSC,WCLPRICE,MEDPRICE,KAMA,MA,TRIMA,WMA,TEMA,EMA,BBANDS"
predictor.setFeatureNames(nameStr)
predictor.setLookback(50)
predictor.classifyMinBar(1.62, 1.63, 1.61, 1.62,80)