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
from libpasteurize.fixes.feature_base import Feature
from tensorflow.python.framework.test_ops import reserved_attr

BINOM_FUNC = owls.binom_logpdf
GLOBAL_PROB_PERIOD=1440*15  # two WEEK
FEATURE_SIZE_DICT = {}
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
        
        self.lastFeature = None
        
        self.unlabeledID = np.array([])
        return
        
    def setInitMin(self,minbar):
        self.initMin = parser.parse(minbar) 
        return 
    
    def loadMinBars(self,barFile):
        self.allMinBars = pd.read_csv(barFile)
        # drop last bar which is not accurate
        self.allMinBars = self.allMinBars.iloc[:-1,:]
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
#           
        self.open = self.allMinBars['OPEN'].values
        self.high = self.allMinBars['HIGH'].values
        self.low  = self.allMinBars['LOW'].values
        self.close = self.allMinBars['CLOSE'].values
        self.tickVol = self.allMinBars['TICKVOL'].values.astype(float)
        self.labels = self.allMinBars['LABEL'].values
        
        self.time = self.allMinBars['TIME'].values
        print "Latest min bar in history: " + self.time[-1]
        return self.time[-1]
    
    def resetFeatureTable(self):
        self.rawFeatures = pd.DataFrame()
        return
    
    def showHistoryMinBars(self):
#         for i in range(len(self.close)):
#             print self.open[i],self.high[i],self.low[i],self.close[i],self.tickVol[i]
        print "Total minbars: %d" % len(self.open) 
        return
    
    def appendNewBar(self,open,high,low,close,tickvol,timestr):
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
        
#         self.labels = np.append(self.labels,-1)
        self.time = np.append(self.time,timestr)
        
#         self.unlabeledID = np.append(self.unlabeledID, len(self.labels)-1)
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
#         f = np.around(f,6)
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
    def setMALookback(self,malookback):
        self.maLookback = malookback
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
            'BMFFT': self.compBMFFT,
            'ROR' : self.compROR,
            'LWMA': self.compLWMA,
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
            
        self.lastFeature = self.rawFeatures.iloc[-1,:].values
    
    def removeNullID(self,ind):
        
        nullID = np.where(np.isnan(ind))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
        return

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
    
        FEATURE_SIZE_DICT['BMFFT'] = data.shape[1]
        
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
       
        tr,ts = owls.compLastBinom(self.labels.astype(float),self.lookback,GLOBAL_PROB_PERIOD)

        return tr,ts
    
    def compBinomial(self):
        Log(LOG_INFO) << "Computing binomial prob ..."
        sells,res = owls.compBinom(self.labels.astype(float),self.lookback,GLOBAL_PROB_PERIOD)
        
        self.removeNullID(res) 
        self.removeNullID(sells)
        
        self.rawFeatures['SELLS'] = sells
        self.rawFeatures['BINOM'] = res
        
        FEATURE_SIZE_DICT['BINOM'] = 2
        Log(LOG_INFO) << "binom done"
        
    def __compBinomial(self):
        Log(LOG_INFO) << "Computing binomial prob..."
        
        if self.binomProb is None:
            self.binomProb = self.compBinomProb()
            
        p = self.binomProb
        
        res=[]
        sells = []
        for i in range(len(self.labels)):
            S = i - GLOBAL_PROB_PERIOD
            if S < 0:
                res.append(np.nan)
                sells.append(np.nan)
                continue
            
            s = i - self.lookback
            arr = self.labels[s:i]
            k = int(sum(arr)) # incorrect if label == -1

            Arr = self.labels[i-GLOBAL_PROB_PERIOD:i]
            a = np.where(Arr==1)[0]
            p = len(a)*1./GLOBAL_PROB_PERIOD
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
        
        FEATURE_SIZE_DICT['BINOM'] = 2
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
        
        FEATURE_SIZE_DICT['ILS'] = 2
            
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
    
    def compROR(self):
        # rate of range
        Log(LOG_INFO) << "Computing ROR ..."
        res =  np.zeros(len(self.close))
        for i in range(len(self.close)):
            if i - self.lookback < 0: 
                res[i] = np.nan
                continue
            h = self.high[i-self.lookback:i]
            l = self.low[i-self.lookback:i]
            uh = np.max(h)
            ul = np.min(l)
            cur= (self.high[i] + self.low[i])*.5;
            res[i] = (cur - ul)*1./(uh - ul)
        
        self.removeNullID(res)
        self.rawFeatures['ROR'] = res
        
        FEATURE_SIZE_DICT['ROR'] = 1
        
        Log(LOG_INFO) << "ROR done"
        return   
                
    def compLWMA(self):
        period=self.maLookback
        Log(LOG_INFO) << "Start LWMA with period %d" % period
        signal = (self.high + self.low)*.5 
        buffer = [np.nan] * period;
        for i in range(period,len(signal)):
            buffer.append(
                (signal[i-period:i] * (np.arange(period)+1)).sum()
                / (np.arange(period)+1).sum()
                )
        bf = np.array(buffer)
        
        res = signal - bf
        self.removeNullID(res)
        self.rawFeatures['LWMA_D'] = res
        
        grad = np.zeros(len(res))
        for i in range(len(res)):
            if i==0:
                grad[i] = np.nan 
                continue
            grad[i] = bf[i] - bf[i-1]
                
        self.removeNullID(grad)
        self.rawFeatures['GRAD_LWMA'] = grad
        
        FEATURE_SIZE_DICT['LWMA'] = 2
        
        Log(LOG_INFO) << "LWMA done"
        return
        
    def compULTOSC(self):
        uc = talib.ULTOSC(self.high,self.low,self.close,timeperiod1=self.lookback/3,
                          timeperiod2=self.lookback/2, timeperiod3=self.lookback)
        self.removeNullID(uc)
        self.rawFeatures['ULTOSC'] = uc
        
        FEATURE_SIZE_DICT['ULTOSC']=1
        return
    
    def compMACDFIX(self):
        macd,ms,mc = talib.MACDFIX(self.close,signalperiod=self.lookback)
        self.removeNullID(macd)
        self.rawFeatures['MACD'] = macd
        
        FEATURE_SIZE_DICT['MACD'] = 1
        return
    
    def compEMA(self):
        ema = talib.EMA(self.close,timeperiod=self.lookback)
        self.removeNullID(ema)
        self.rawFeatures['EMA']=ema
        
        FEATURE_SIZE_DICT['EMA'] = 1
        return
    
    def compTEMA(self):
        tema = talib.TEMA(self.close,timeperiod=self.lookback)
        self.removeNullID(tema)
        self.rawFeatures['TEMA'] = tema 
        
        FEATURE_SIZE_DICT['TEMA'] = 1
        return
    
    def compWMA(self):
        wma = talib.WMA(self.close,timeperiod=self.lookback)
        self.removeNullID(wma)
        self.rawFeatures['WMA'] = wma 
        
        FEATURE_SIZE_DICT['WMA'] = 1
        return
    
    def compTRIMA(self):
        ta = talib.TRIMA(self.close,timeperiod=self.lookback)
        self.removeNullID(ta)
        self.rawFeatures['TRIMA'] = ta 
        
        FEATURE_SIZE_DICT['TRIMA'] =1
        return
    
    def compMA(self):
        ma = talib.MA(self.close,timeperiod=self.lookback)
        self.removeNullID(ma)
        self.rawFeatures['MA']=ma 
        
        FEATURE_SIZE_DICT['MA'] = 1
        return
    
    def compMEDPRICE(self):
        med = talib.MEDPRICE(self.high,self.low)
        self.removeNullID(med)
        self.rawFeatures['MEDPRICE'] = med 
        
        FEATURE_SIZE_DICT['MEDPRICE'] = 1
        return
    
    def compWCLPRICE(self):
        wcl = talib.WCLPRICE(self.high,self.low,self.close)
        self.removeNullID(wcl)
        self.rawFeatures['WCLPRICE'] = wcl 
        
        FEATURE_SIZE_DICT['WCLPRICE'] = 1
        return
        
    def compOBV(self):
        obv  = talib.OBV(self.close,self.tickVol)
        self.removeNullID(obv)
        self.rawFeatures['OBV'] = obv
        
        FEATURE_SIZE_DICT['OBV'] = 1
        return
    
    def compAD(self):
        ad = talib.AD(self.high,self.low,self.close,self.tickVol)
        self.removeNullID(ad)
        self.rawFeatures['AD'] = ad 
        
        FEATURE_SIZE_DICT['AD'] = 1
        return
    
    def compPLUSDI(self):
        pdi = talib.PLUS_DI(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(pdi)
        self.rawFeatures['PLUS_DI']=pdi 
        
        FEATURE_SIZE_DICT['PLUS_DI'] = 1
        return
    
    def compPLUSDM(self):
        pdm = talib.PLUS_DM(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(pdm)
        self.rawFeatures['PLUS_DM'] = pdm
        
        FEATURE_SIZE_DICT['PLUS_DM'] = 1
        return
    def compMOM(self):
        mom = talib.MOM(self.close,timeperiod=self.lookback)
        self.removeNullID(mom)
        self.rawFeatures['MOM'] = mom 
        
        FEATURE_SIZE_DICT['MOM'] = 1
        return
    
    def compMINUSDI(self):
        mi = talib.MINUS_DI(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(mi)
        self.rawFeatures['MINUS_DI'] = mi 
        
        FEATURE_SIZE_DICT['MINUS_DI'] = 1
        return
    
    def compMINUSDM(self):
        md = talib.MINUS_DM(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(md)
        self.rawFeatures['MINUS_DM'] = md 
        
        FEATURE_SIZE_DICT['MINUS_DM'] = 1
        return
    
    def compMFI(self):
        mfi = talib.MFI(self.high,self.low,self.close,self.tickVol,timeperiod=self.lookback)
        self.removeNullID(mfi)
        self.rawFeatures['MFI'] = mfi
        
        FEATURE_SIZE_DICT['MFI'] = 1
        return
    
    def compNATR(self):
        natr = talib.NATR(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(natr)
        self.rawFeatures['NATR'] = natr 
        
        FEATURE_SIZE_DICT['NATR'] = 1
        return
    
    def compMidPrice(self):
        mp = talib.MIDPRICE(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(mp)
        
        self.rawFeatures['MIDPRICE'] = mp
        
        FEATURE_SIZE_DICT['MIDPRICE'] = 1
        return
    
    def compKAMA(self):
        kama = talib.KAMA(self.close,timeperiod=self.lookback)
        self.removeNullID(kama)
        
        self.rawFeatures['KAMA'] = kama
        
        FEATURE_SIZE_DICT['KAMA'] = 1
        return
    
    def __compRSI(self):
        rsi = talib.RSI(self.close,timeperiod=self.lookback)
        self.removeNullID(rsi)
        self.rawFeatures['RSI'] = rsi
        
        FEATURE_SIZE_DICT['RSI'] = 1
        return
    
    def compRSI(self):
        Log(LOG_INFO) << "Owls RSI ...%d %d %d" % (len(self.open),len(self.close),self.lookback)
        up,down,rsi = owls.compRSI(self.open,self.close,self.lookback,1e-5)
        self.removeNullID(rsi)
        self.rawFeatures['NET_UP'] = up 
        self.rawFeatures['DOWN'] = down
        self.rawFeatures['RSI'] = rsi
        
        FEATURE_SIZE_DICT['RSI'] = 3
        
        Log(LOG_INFO) << "Owls RSI done"
        return
    
    def compWILLR(self):
        wr = talib.WILLR(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(wr)
        self.rawFeatures['WILLR'] = wr
        
        FEATURE_SIZE_DICT['WILLR'] = 1
        return
    
    def compTRIX(self):
        tx = talib.TRIX(self.close,timeperiod=self.lookback)
        self.removeNullID(tx)
        self.rawFeatures['TRIX'] = tx
        
        FEATURE_SIZE_DICT['TRIX'] = 1
        return
    
    def compROC(self):
        roc = talib.ROC(self.close,timeperiod=self.lookback)
        self.removeNullID(roc)
        self.rawFeatures['ROC'] = roc 
        
        FEATURE_SIZE_DICT['ROC'] = 1
        return
    
    def compAROONOSC(self):
        ac = talib.AROONOSC(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(ac)
        self.rawFeatures['AROONOSC'] = ac
        
        FEATURE_SIZE_DICT['AROONOSC'] = 1
        return
    
    def compADX(self):
        adx = talib.ADX(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(adx)
        self.rawFeatures['ADX'] = adx 
        
        FEATURE_SIZE_DICT['ADX']=1
        return
    
    def compATR(self):
        atr = talib.ATR(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(atr)
        self.rawFeatures['ATR'] = atr 
        
        FEATURE_SIZE_DICT['ATR'] = 1
        return
    
    def compDX(self):
        dx = talib.DX(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(dx)
        self.rawFeatures['DX'] = dx
        
        FEATURE_SIZE_DICT['DX'] = 1
        return
    
    def compTSF(self):
        tsf = talib.TSF(self.close,timeperiod=self.lookback) 
        self.removeNullID(tsf)
        self.rawFeatures['TSF'] = tsf
        
        FEATURE_SIZE_DICT['TSF'] = 1
        
        return
    
    def compCMO(self):
        cmo = talib.CMO(self.close,timeperiod=self.lookback)
        self.removeNullID(cmo)
        self.rawFeatures['CMO'] = cmo 
        
        FEATURE_SIZE_DICT['CMO'] = 1
        return
    
    def compBETA(self):
        beta = talib.BETA(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(beta)
        self.rawFeatures['BETA'] = beta
        
        FEATURE_SIZE_DICT['BETA'] = 1
        return
    
    def compBBANDS(self):
        ub,mb,lb = talib.BBANDS(self.close,timeperiod=self.lookback)
        self.removeNullID(ub)
        self.rawFeatures['UPPERBAND'] = ub 
        self.rawFeatures['MIDDLEBAND'] = mb
        self.rawFeatures['LOWERBAND'] = lb 

        FEATURE_SIZE_DICT['BBANDS'] = 3
        return
    
    def compCCI(self):
        cci = talib.CCI(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(cci)
        self.rawFeatures['CCI']=cci 
        
        FEATURE_SIZE_DICT['CCI'] = 1
        return
    
    def getTotalFeatureMatrix(self,isDropN1=True):
        lz = len(self.nullID)+1
        data = self.rawFeatures.values[lz:,:]

        labels = self.labels[lz:]
        time = self.time[lz:]
            
        data = data[:-1,:]
        labels = labels[1:]

        time = time[1:]
        
        self.labels = self.labels[lz:]
        self.open = self.open[lz:]
        self.high = self.high[lz:]
        self.low  = self.low[lz:]
        self.close = self.close[lz:]
        self.time = self.time[lz:]
        
        df = pd.DataFrame(data,columns=self.rawFeatures.keys())
        
        df.insert(0,'label',labels)
        df.insert(0,'time',time)
        
        # remove label == -1
        if len(np.where(labels==-1)[0])>0 and isDropN1:
            ids = list(np.where(labels==-1)[0])
            idx = ids[0]
            df=df.iloc[:idx,:]
#             df = df.drop(df.index[ids])
                
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
    
    def predictUnlabeledBars(self,model,lookback):

        fm,labels = self.getTotalFeatureMatrix(isDropN1=False)
        unlabeled = np.where(labels<0)[0]
        
        self.unlabeledID = unlabeled
        
#         print self.unlabeledID
        
        print "Unlabeled hist bars: %d" % len(unlabeled)
#         print fm.shape[0],len(labels)
#         print fm
        for i in unlabeled:
            arr = labels[i - GLOBAL_PROB_PERIOD-1:i]
            tr,ts = owls.compLastBinom(arr.astype(float),lookback,GLOBAL_PROB_PERIOD)
            
            f = fm[i,:]
            f[-2] = tr
            f[-1] = ts        
            
            labels[i] = model.predict(f.reshape(1,-1))[0]
         
        self.labels = labels.astype(int)
        
        ### Compute target of the last feature, which is removed when getting total features

        arr = self.labels[- GLOBAL_PROB_PERIOD-1:]
        tr,ts = owls.compLastBinom(arr.astype(float),lookback,GLOBAL_PROB_PERIOD)

        self.lastFeature[-2] = tr 
        self.lastFeature[-1] = ts
        lb = model.predict(self.lastFeature.reshape(1,-1))[0]
        self.labels = np.append(self.labels,lb)
        self.unlabeledID = np.append(self.unlabeledID, len(self.labels)-1)

        return
             
    def appendLatestLabel(self,pred):
        self.labels = np.append(self.labels, pred)
        self.unlabeledID = np.append(self.unlabeledID,len(self.labels)-1)

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
        
        if len(self.unlabeledID) == 0:
            print "No past labels need to be corrected"
            return
        else:
            print "%d past labels need to be corrected" % len(self.unlabeledID)
        
        high = self.high[-1]
        low  = self.low[-1]
        
        correctedID=[]
        for i in range(len(self.unlabeledID)):
            idx = int(self.unlabeledID[i])
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
        
        print "Marking done"    
        if len(correctedID) > 0:
            self.unlabeledID =  np.delete(self.unlabeledID, correctedID)
            print "History bars marked: %d" % len(correctedID)

            
            
    
    