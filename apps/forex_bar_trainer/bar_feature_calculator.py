'''
Created on Dec 2, 2018

@author: fxua
'''
import talib 
import numpy as np
import pandas as pd
import pdb
from modules.basics.common.logger import *
from dateutil import parser
from modules.basics.common.utils import smooth1D
from scipy.stats import binom
import owls
class BarFeatureCalculator(object):
    '''
    classdocs
    '''


    def __init__(self, config=None):
        '''
        Constructor
        '''
        self.config = config
        self.initMin = None
        self.rawFeatures = pd.DataFrame()
        self.nullID = np.array([])
        self.close = np.array([])
        self.open  = np.array([])
        self.high  = np.array([])
        self.low   = np.array([])
        self.tickVol= np.array([])
        self.binomProb = None
        return
        
    def setInitMin(self,minbar):
        self.initMin = parser.parse(minbar) 
        return 
    
    def loadMinBars(self,barFile):
        self.allMinBars = pd.read_csv(barFile)
        N = self.allMinBars.shape[0]
        print "Hisotry min bars loaded: " + str(N)
        
        if self.initMin is not None:
            k = N-1
            while k >= 0:
                t = parser.parse(self.allMinBars.iloc[k,:]['TIME'])
                dt = self.initMin - t
                if dt.total_seconds() > 0:
                    print "searching init min bar done: " + str(k)
                    
                    break
                k-=1
                
            if k < N-1:
                self.allMinBars = self.allMinBars.iloc[:k+2,:]

        Log(LOG_INFO) << "Latest min bar in history: " + self.allMinBars.iloc[-1,:]['TIME']  
        
#         if self.initMin is not None:
#             self.allMinBars = self.allMinBars.iloc[-50000:,:]
#           
        self.open = self.allMinBars['OPEN']
        self.high = self.allMinBars['HIGH']
        self.low  = self.allMinBars['LOW']
        self.close = self.allMinBars['CLOSE']
        self.tickVol = self.allMinBars['TICKVOL']
        self.labels = self.allMinBars['LABEL'].values
        
        self.time = self.allMinBars['TIME'].values
        print self.time[-1]
        return self.time[-1]
    
    def resetFeatureTable(self):
        self.rawFeatures = pd.DataFrame()
        return
    
    def showHistoryMinBars(self):
#         for i in range(len(self.close)):
#             print self.open[i],self.high[i],self.low[i],self.close[i],self.tickVol[i]
        print "Total minbars: %d" % len(self.open) 
        return
    
    def appendNewBar(self,open,high,low,close,tickvol):
        open=np.around(open,5)
        high=np.around(high,5)
        low = np.around(low,5)
        close=np.around(close,5)
        tickvol=np.around(tickvol,0)
        
        self.open = np.append(self.open, open)
        self.high = np.append(self.high,high)
        self.low  = np.append(self.low,low)
        self.close= np.append(self.close,close)
        self.tickVol = np.append(self.tickVol,tickvol)
        
        self.labels = np.append(self.labels,-1)
        return
    
    def getLatestFeatures(self):
#         print self.rawFeatures
        
        f = self.rawFeatures.iloc[-1,:].values
        f = np.around(f,6)
#         print self.rawFeatures.values

        allbars = np.vstack([self.open,self.high,self.low,self.close,self.tickVol])
        allbars= allbars.transpose()
        print allbars[-10:,:]  
#         
#         df = pd.DataFrame(data=allbars,index=False)
#         df.to_csv('hist_bars.csv')                   
        return f
    
    def getLatestMinBar(self):
        mb = self.allMinBars.iloc[-1,:]
        return mb
    
    def setLookback(self,lookback):
        self.lookback = lookback
        return
    
    def computeFeatures(self,featureNames,latestBars=None):
        BarFeatureSwitcher = {
            "MIDPRICE": self.compMidPrice,
            "KAMA" : self.compKAMA,
            "RSI" : self.compRSI,
            "WILLR" : self.compWILLR,
            "TRIX" : self.compTRIX,
            "ROC" : self.compROC,
            "AROONOSC" : self.compAROONOSC,
            "ADX" : self.compADX,
            "DX" : self.compDX,
            "CMO" : self.compCMO,
            "BETA" : self.compBETA,
            "BBANDS" : self.compBBANDS,
            "ATR": self.compATR,
            "CCI": self.compCCI,
            "NATR" : self.compNATR,
            "TSF" : self.compTSF,
            "MFI" : self.compMFI,
            "MINUS_DI": self.compMINUSDI,
            "MINUS_DM" : self.compMINUSDM,
            "MOM" : self.compMOM,
            "PLUS_DI" : self.compPLUSDI,
            "PLUS_DM" : self.compPLUSDM,
            "AD" : self.compAD,
            "OBV" : self.compOBV,
            "WCLPRICE" : self.compWCLPRICE,
            "MEDPRICE" : self.compMEDPRICE,
            "MA" : self.compMA,
            "TRIMA" : self.compTRIMA,
            "WMA" : self.compWMA,
            'TEMA' : self.compTEMA,
            "EMA" : self.compEMA,
            'MACDFIX': self.compMACDFIX,
            'ULTOSC': self.compULTOSC,
            'ILS': self.compILS,
            'BINOM': self.compBinomial
        }
        
        if latestBars is None:
            latestBars = len(self.open)
            
        self.open = np.around(self.open,5)[-latestBars:]
        self.high = np.around(self.high,5)[-latestBars:]
        self.low  = np.around(self.low,5)[-latestBars:]
        self.close= np.around(self.close,5)[-latestBars:]
        self.tickVol = np.around(self.tickVol,0)[-latestBars:]
        
        self.labels = self.labels[-latestBars:]
        self.time = self.time[-latestBars:]
        
        for fn in featureNames:
            BarFeatureSwitcher[fn]()
    
    def removeNullID(self,ind):
        
        nullID = np.where(np.isnan(ind))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
        return
    
    def getBinomProb(self):
        return self.binomProb
    
    def setBinomProb(self,pb):
        self.binomProb = pb
        return
    
    def compBinomProb(self):
        label = self.labels
        n_tbd = len(np.where(label==-1)[0])
        n_sell = len(np.where(label==1)[0])
        p = n_sell*1./(len(label)-n_tbd)
        
        return p
    
    def compLatestBinom(self):
        arr = self.labels[-self.lookback-1:-1]
        k = sum(arr)
        pb = owls.binom_entropy(k+1,self.lookback+1,self.binomProb)
        
        return k*1./self.lookback,pb
    
    def compBinomial(self):
        Log(LOG_INFO) << "Computing binomial prob..."
#         label = self.labels
#         n_tbd = len(np.where(label==-1)[0])
#         n_sell = len(np.where(label==1)[0])
#         p = n_sell*1./(len(label)-n_tbd)

        
        if self.binomProb is None:
            self.binomProb = self.compBinomProb()
            
        p = self.binomProb
        
        Log(LOG_INFO) << "Sell prob: %f" % p
        res=[]
        sells = []
        for i in range(len(self.labels)):
            s = i-self.lookback 
            if s < 0:
                res.append(np.nan)
                sells.append(np.nan)
                continue
            arr = self.labels[s:i]
            k = sum(arr) # incorrect if label == -1
#             pb = binom.pmf(k+1,self.lookback+1,p)
            pb = owls.binom_entropy(k+1,self.lookback+1,p)
            res.append(pb)
            sells.append(k*1./self.lookback)
           
        res = np.array(res)
        sells = np.array(sells)
        self.removeNullID(res) 
        self.removeNullID(sells)
        self.rawFeatures['SELLS'] = sells
        self.rawFeatures['BINOM'] = res
        
        Log(LOG_INFO) << "binom done"
        return
            
    def compILS(self):
        mp = talib.MEDPRICE(self.high,self.low)
        mmp = talib.SMA(mp,timeperiod=10)
        dmmp = np.diff(mmp)
        dmmp = np.insert(dmmp,0,np.nan)
        
        ddmmp = np.diff(dmmp)
        ddmmp = np.insert(ddmmp,0,np.nan)
        
        ils = dmmp/mmp*1e3
        
        ave = talib.SMA(ils,timeperiod=30)
        
        self.removeNullID(ils)
        self.removeNullID(ave)
#         self.removeNullID(ddmmp)
   
        self.rawFeatures['ILS'] = ils
        self.rawFeatures['MA_ILS'] = ave
            
        ## find nearest turning 
#         wl=11
#         smp = smooth1D(mp, window_len=wl, window='hanning')
#         smp = smp[wl-1:]
# #         smp = talib.SMA(mp,timeperiod=60)
#         dsmp = np.diff(smp)
#         dsmp = np.insert(dsmp,0,np.nan)
#         sg = np.sign(dsmp)
#         
#         tds =  np.zeros(len(dsmp))       
#         for i in range(len(sg)):
#             s = sg[i]
#             for k in range(10000):
#                 if i-k<=0:
#                     tds[i] = np.nan
#                     break
#                 if sg[i-k] != s:
#                     tds[i] = k
#                     break
#                 
# #         pdb.set_trace()
#         self.removeNullID(tds)
#         self.rawFeatures['TURN_DIS']=tds
#         
        return
        
    def compULTOSC(self):
        uc = talib.ULTOSC(self.high,self.low,self.close,timeperiod1=self.lookback/3,
                          timeperiod2=self.lookback/2, timeperiod3=self.lookback)
        self.removeNullID(uc)
        self.rawFeatures['ULTOSC'] = uc
        return
    
    def compMACDFIX(self):
        macd,ms,mc = talib.MACDFIX(self.close,signalperiod=self.lookback)
        self.removeNullID(macd)
        self.rawFeatures['MACD'] = macd
        return
    
    def compEMA(self):
        ema = talib.EMA(self.close,timeperiod=self.lookback)
        self.removeNullID(ema)
        self.rawFeatures['EMA']=ema
        return
    
    def compTEMA(self):
        tema = talib.TEMA(self.close,timeperiod=self.lookback)
        self.removeNullID(tema)
        self.rawFeatures['TEMA'] = tema 
        return
    
    def compWMA(self):
        wma = talib.WMA(self.close,timeperiod=self.lookback)
        self.removeNullID(wma)
        self.rawFeatures['WMA'] = wma 
        return
    
    def compTRIMA(self):
        ta = talib.TRIMA(self.close,timeperiod=self.lookback)
        self.removeNullID(ta)
        self.rawFeatures['TRIMA'] = ta 
        return
    
    def compMA(self):
        ma = talib.MA(self.close,timeperiod=self.lookback)
        self.removeNullID(ma)
        self.rawFeatures['MA']=ma 
        return
    
    def compMEDPRICE(self):
        med = talib.MEDPRICE(self.high,self.low)
        self.removeNullID(med)
        self.rawFeatures['MEDPRICE'] = med 
        return
    
    def compWCLPRICE(self):
        wcl = talib.WCLPRICE(self.high,self.low,self.close)
        self.removeNullID(wcl)
        self.rawFeatures['WCLPRICE'] = wcl 
        return
        
    def compOBV(self):
        obv  = talib.OBV(self.close,self.tickVol)
        self.removeNullID(obv)
        self.rawFeatures['OBV'] = obv
        return
    
    def compAD(self):
        ad = talib.AD(self.high,self.low,self.close,self.tickVol)
        self.removeNullID(ad)
        self.rawFeatures['AD'] = ad 
        return
    
    def compPLUSDI(self):
        pdi = talib.PLUS_DI(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(pdi)
        self.rawFeatures['PLUS_DI']=pdi 
        return
    
    def compPLUSDM(self):
        pdm = talib.PLUS_DM(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(pdm)
        self.rawFeatures['PLUS_DM'] = pdm
        return
    def compMOM(self):
        mom = talib.MOM(self.close,timeperiod=self.lookback)
        self.removeNullID(mom)
        self.rawFeatures['MOM'] = mom 
        return
    
    def compMINUSDI(self):
        mi = talib.MINUS_DI(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(mi)
        self.rawFeatures['MINUS_DI'] = mi 
        return
    
    def compMINUSDM(self):
        md = talib.MINUS_DM(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(md)
        self.rawFeatures['MINUS_DM'] = md 
        return
    
    def compMFI(self):
        mfi = talib.MFI(self.high,self.low,self.close,self.tickVol,timeperiod=self.lookback)
        self.removeNullID(mfi)
        self.rawFeatures['MFI'] = mfi
        return
    
    def compNATR(self):
        natr = talib.NATR(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(natr)
        self.rawFeatures['NATR'] = natr 
        return
    
    def compMidPrice(self):
        mp = talib.MIDPRICE(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(mp)
        
        self.rawFeatures['MIDPRICE'] = mp
        return
    
    def compKAMA(self):
        kama = talib.KAMA(self.close,timeperiod=self.lookback)
        self.removeNullID(kama)
        
        self.rawFeatures['KAMA'] = kama
        return
    
    def compRSI(self):
        rsi = talib.RSI(self.close,timeperiod=self.lookback)
        self.removeNullID(rsi)
        self.rawFeatures['RSI'] = rsi
        return
    
    def compWILLR(self):
        wr = talib.WILLR(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(wr)
        self.rawFeatures['WILLR'] = wr
        return
    
    def compTRIX(self):
        tx = talib.TRIX(self.close,timeperiod=self.lookback)
        self.removeNullID(tx)
        self.rawFeatures['TRIX'] = tx
        return
    
    def compROC(self):
        roc = talib.ROC(self.close,timeperiod=self.lookback)
        self.removeNullID(roc)
        self.rawFeatures['ROC'] = roc 
        return
    
    def compAROONOSC(self):
        ac = talib.AROONOSC(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(ac)
        self.rawFeatures['AROONOSC'] = ac
        return
    
    def compADX(self):
        adx = talib.ADX(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(adx)
        self.rawFeatures['ADX'] = adx 
        return
    
    def compATR(self):
        atr = talib.ATR(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(atr)
        self.rawFeatures['ATR'] = atr 
        return
    
    def compDX(self):
        dx = talib.DX(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(dx)
        self.rawFeatures['DX'] = dx
        return
    
    def compTSF(self):
        tsf = talib.TSF(self.close,timeperiod=self.lookback) 
        self.removeNullID(tsf)
        self.rawFeatures['TSF'] = tsf
        return
    
    def compCMO(self):
        cmo = talib.CMO(self.close,timeperiod=self.lookback)
        self.removeNullID(cmo)
        self.rawFeatures['CMO'] = cmo 
        return
    
    def compBETA(self):
        beta = talib.BETA(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(beta)
        self.rawFeatures['BETA'] = beta
        return
    
    def compBBANDS(self):
        ub,mb,lb = talib.BBANDS(self.close,timeperiod=self.lookback)
        self.removeNullID(ub)
        self.rawFeatures['UPPERBAND'] = ub 
        self.rawFeatures['MIDDLEBAND'] = mb
        self.rawFeatures['LOWERBAND'] = lb 

        return
    
    def compCCI(self):
        cci = talib.CCI(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(cci)
        self.rawFeatures['CCI']=cci 
        return
    
    def getTotalFeatureMatrix(self,isDropN1=True):
        data = self.rawFeatures.values[len(self.nullID)+1:,:]
        data = np.around(data,6)
        labels = self.labels[len(self.nullID)+1:]
        self.time = self.time[len(self.nullID)+1:]
        
        sells = self.rawFeatures['SELLS'].values[len(self.nullID)+1:]
        binom = self.rawFeatures['BINOM'].values[len(self.nullID)+1:]
        data = data[:-1,:]
        labels = labels[1:]
        binom = binom[1:]
        sells = sells[1:]
        self.time = self.time[1:]
        
        df = pd.DataFrame(data,columns=self.rawFeatures.keys())
        
        df.insert(0,'label',labels)
        df.insert(0,'time',self.time)
        
        df['SELLS']=sells
        df['BINOM']=binom
        # remove label == -1
        if len(np.where(labels==-1)[0])>0 and isDropN1:
            ids = list(np.where(labels==-1)[0])
#             df=df.iloc[:idx,:]
            df = df.drop(df.index[ids])
                
        df.to_csv("features.csv",index=False)
        
        Log(LOG_INFO) << "Feature file dumped: features.csv"
        if data.shape[0] != len(labels):
            Log(LOG_FATAL) << "Samples inconsistent with labels"
            
        ks = list(self.rawFeatures.keys())
        data = df[ks].values
        labels = df['label'].values
        return data,labels
    
    def getTime(self):
        return self.time
    
    def labelUnlabeledBars(self,model,lookback):
        p = self.getBinomProb()
        fm,labels = self.getTotalFeatureMatrix(isDropN1=False)
        unlabeled = np.where(labels<0)[0]
        
        print "Unlabeled hist bars: %d" % len(unlabeled)
        for id in unlabeled:
            arr = labels[id - lookback:id]
            k = sum(arr)
            
            pb = owls.binom_entropy(k+1,lookback+1,p)
            
            f = fm[id,:]
            f[-2] = k*1./lookback
            f[-1] = pb
            labels[id] = model.predict(f.reshape(1,-1))[0]
         
        self.labels = labels   
        return
             
    def setLatestLabel(self,pred):
        self.labels[-1] = pred
        return
            
        
    
    