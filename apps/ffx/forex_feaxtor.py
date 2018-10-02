'''
Created on Oct 2, 2018

@author: fxua
'''
from apps.app import App
from apps.ffx.ffxconf import FFXConfig
import csv
from dateutil import parser 
ONEMIN = 60
HALFMIN = 30
class ForexFex(App):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(ForexFex,self).__init__()
        self.config = FFXConfig()
        self.allTicks = []
        self.buyTicks = []
        self.sellTicks = []
        return
    
    def loadTickFile(self):
        with open(self.config.getTickFile(),'r') as f:
            reader = csv.reader(f,delimiter='\t')
            for row in reader:
                if '<DATE>' in row:
                    continue
                sample={}
                timestr = row[0] + " " + row[1]
                t = parser.parse(timestr)
                                
                sample['time'] = t
                bid = row[2]
                if bid == "":
                    sample['bid'] = None
                else:
                    sample['bid'] = float(bid)
                    
                ask = row[3]
                if ask == "":
                    sample['ask'] = None
                else:
                    sample['ask'] = float(ask)
                    
                self.allTicks.append(sample)
        return
        
    def _extractValidTicks(self,opt='ask'):
        ticks=[]
        prev = None
        for sample in self.allTicks:
            t = sample['time']
            if prev is None and sample[opt] is not None:
                prev = t
                ticks.append(sample)

            dt = t - prev
            if dt.total_seconds() < HALFMIN or sample[opt] is None:
                continue
            ticks.append(sample)
            prev = t
        return ticks
    
    def makeBuyLabel(self):
            
    def prepare(self):
        self.loadTickFile()
        self.buyTicks = self._extractValidTicks('ask')
        self.sellTicks = self._extractValidTicks('bid')
        return