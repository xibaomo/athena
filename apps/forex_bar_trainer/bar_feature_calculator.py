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

BINOM_FUNC = owls.binom_logpdf

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
        
        self.unlabeledID = np.array([])
        return
        
    def setInitMin(self,minbar):
        self.initMin = parser.parse(minbar) 
        return 
    
    def loadMinBars(self,barFile):
        self.allMinBars = pd.read_csv(barFile)
        N = self.allMinBars.shape[0]
        print "Hisotry min bar file loaded: " + str(N)
        
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
        print "Latest min bar: " + self.time[-1]
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
        self.time = np.append(self.time,'unknown_time')
        
        self.unlabeledID = np.append(self.unlabeledID, len(self.labels)-1)
        return
    
    def loadHistoryMinBars(self,data):
        print "loading history min bars ..."
        self.open = np.append(self.open, np.around(data[:,0],5))
        self.high = np.append(self.high, np.around(data[:,1],5))
        self.low = np.append(self.low, np.around(data[:,2],5))
        self.close = np.append(self.close,np.around(data[:,3],5))
        self.tickVol = np.append(self.tickVol,data[:,4])
        
        newtime = ["unknown_time" for x in range(data.shape[0])]
        newlabels = np.ones(data.shape[0]) * (-1)
        
        self.labels = np.append(self.labels,newlabels)
        
        self.time = np.append(self.time,newtime)
        
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
    
    def computeFeatures(self,featureNames,latestBars=None, lastOnly = False):
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
            'BINOM': self.compBinomial,
            'BMFFT': self.compBMFFT
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
#             print "feature: " + fn
            BarFeatureSwitcher[fn]()
    
    def removeNullID(self,ind):
        
        nullID = np.where(np.isnan(ind))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
        return
    
    def compLabelFFT(self,labels):
        N=4
#         N=self.lookback/2+1
        if len(labels)  < self.lookback:
            return np.ones(N) * np.nan
        
        ff = np.fft.fft(labels)
        
        f = np.abs(ff[:self.lookback/2+1])
        
        return f[:N]
        
    def compBMFFT(self):
        Log(LOG_INFO) << "Computing BMFFT ..."
        data = []
        
        for i in range(len(self.labels)):
#             print i
            s = i - self.lookback + 1
            if s < 0:
                s = 0
            labels = self.labels[s:i+1]
            f = self.compLabelFFT(labels)
            
            data.append(np.abs(f))
            
        Log(LOG_INFO) <<"BMFFT done. Appending to feature table ..."
        
        data = np.array(data)
        self.removeNullID(data[:,0])
        
        for i in range(data.shape[1]):
            key = "F_" + str(i)
            self.rawFeatures[key] = data[:,i]
    
        Log(LOG_INFO) << "BMFFT feature added"
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
        
        print "Sell prob: %f" % p
        
        return p
    
    def compLatestBinom(self):
#         self.binomProb = self.compBinomProb()
        
        arr = self.labels[-self.lookback-1:-1]
        k = sum(arr)
        pb = BINOM_FUNC(k+1,self.lookback+1,self.binomProb)

        return k*1./self.lookback,pb
    
    def compBinomial(self):
        Log(LOG_INFO) << "Computing binomial prob..."
        
        if self.binomProb is None:
            self.binomProb = self.compBinomProb()
            
        p = self.binomProb
        
        res=[]
        sells = []
        for i in range(len(self.labels)):
                
            s = i-self.lookback 
            if s < 0:
                res.append(np.nan)
                sells.append(np.nan)
                continue
            arr = self.labels[s:i]
            k = int(sum(arr)) # incorrect if label == -1

            pb=-1
            if k>=0:
                pb = BINOM_FUNC(k+1,self.lookback+1,p)
            
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
        time = self.time[len(self.nullID)+1:]
        binom = None
        
        
        if "BINOM" in self.rawFeatures.keys():
            sells = self.rawFeatures['SELLS'].values[len(self.nullID)+1:]
            binom = self.rawFeatures['BINOM'].values[len(self.nullID)+1:]
            binom = binom[1:]
            sells = sells[1:]
            
        data = data[:-1,:]
        labels = labels[1:]

        time = time[1:]
        
        lz = len(self.nullID)+1
        self.labels = self.labels[lz:]
        self.open = self.open[lz:]
        self.high = self.high[lz:]
        self.low  = self.low[lz:]
        self.close = self.close[lz:]
        self.time = self.time[lz:]
        
        df = pd.DataFrame(data,columns=self.rawFeatures.keys())
        
        df.insert(0,'label',labels)
        df.insert(0,'time',time)
        
        if binom is not None:
            df['SELLS']=sells
            df['BINOM']=binom
        # remove label == -1
        if len(np.where(labels==-1)[0])>0 and isDropN1:
            ids = list(np.where(labels==-1)[0])
#             df=df.iloc[:idx,:]
            df = df.drop(df.index[ids])
                
#         df.to_csv("features.csv",index=False)
#         
#         Log(LOG_INFO) << "Feature file dumped: features.csv"
        if data.shape[0] != len(labels):
            Log(LOG_FATAL) << "Samples inconsistent with labels"
            
        ks = list(self.rawFeatures.keys())
        data = df[ks].values
        labels = df['label'].values
        return data,labels
    
    def getTime(self):
        return self.time
    
    def predictUnlabeledBars(self,model,lookback):
        p = self.getBinomProb()
        fm,labels = self.getTotalFeatureMatrix(isDropN1=False)
        unlabeled = np.where(labels<0)[0]
        
        self.unlabeledID = unlabeled
        
#         print self.unlabeledID
        
        print "Unlabeled hist bars: %d" % len(unlabeled)
#         print fm.shape[0],len(labels)
#         print fm
        for i in unlabeled:
            arr = labels[i - lookback:i]
            k = int(sum(arr))
            
            pb = BINOM_FUNC(k+1,lookback+1,p)
            
#             print "pb: %f %f %d" % (pb,p,lookback)
#             print fm.shape,i
#             print model
#             print fm[i,:]
            f = fm[i,:]
#             print f 
            
            f[-2] = k*1./lookback
            f[-1] = pb
            
            
            labels[i] = model.predict(f.reshape(1,-1))[0]
         
        self.labels = labels.astype(int)
        
        print "labels = ",self.labels
        return
             
    def setLatestLabel(self,pred):
        self.labels[-1] = pred

        return
        
    def markUnlabeled(self,tp = 200, spread = 10,digits=1.e-5):
        unlabeledID = np.where(self.labels<0)[0]
        
        print unlabeledID
        
        for id in unlabeledID.tolist():
            ub = self.open[id] + (spread + tp)*digits 
            lb = self.open[id] + (spread - tp)*digits
            s = id + 1
            while s < len(self.labels):
                if self.high[s] >= ub and self.low[s] > lb:
                    self.labels[id] = 0
#                     print "labeled 0: %d" % id 
                    break
                elif self.low[s]  <= lb and self.high[s] < ub:
                    self.labels[id] = 1
#                     print "labeled 1: %d" % id 
                    break
                
                elif self.high[s] >= ub and self.low[s] <= lb:
                    self.labels[id] = 2
                    print "rush hour, high-low = %f" % (self.high[s] -self.low[s])
                    break
                elif self.high[s] < ub and self.low[s] > lb:
                    s+=1
                    continue
                else:
                    print 'impossible minute'
                    print self.high[s],ub,self.low[s],lb
                    
        print "All min bars marked"
        

    def correctPastPrediction(self,tp=200,spread=10,digits=1.e-5):
        high = self.high[-1]
        low  = self.low[-1]
        
        correctedID=[]
        for i in range(len(self.unlabeledID)):
            idx = self.unlabeledID[i]
            price = self.open[idx]
            ub = price + (spread + tp)*digits 
            lb = price + (spread - tp)*digits
            if high >= ub and low > lb:
                self.labels[idx] = 0
                correctedID.append(i)
                print "marked 0 : %s %f %f %f" % (self.time[idx],price,high,high-price)
                
            elif low <= lb and high < ub:
                self.labels[idx] = 1
                correctedID.append(i)
                print "marked 1: %s %f %f %f" % (self.time[idx],price,low,price-low)
                
            elif high >= ub and low <= lb:
                self.labels[idx] = 2
                correctedID.append(i)
            else:
                pass
            
        if len(correctedID) > 0:
            self.unlabeledID =  np.delete(self.unlabeledID, correctedID)
            print "History bars marked: %d" % len(correctedID)
            
            
    
    