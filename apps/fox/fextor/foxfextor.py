'''
Created on Oct 1, 2018

@author: fxua
'''
from modules.feature_extractor.extractor import FeatureExtractor
from apps.fox.fextor.foxfexconf import FoxFexConfig
import csv
from dateutil import parser
ONEMIN=60
HALFMIN=30

class FoxFextor(FeatureExtractor):
    '''
    classdocs
    '''


    def __init__(self,foxconfig):
        '''
        Constructor
        '''
        self.config = FoxFexConfig(foxconfig)
        self.allTicks = []
        return
    
    def loadTickFile(self):
        prev=None
        with open(self.config.getTickFile(),'r') as f:
            reader = csv.reader(f,delimiter='\t')
            for row in reader:
                if '<DATE>' in row:
                    continue
                sample=[]
                timestr = row[0] + " " + row[1]
                t = parser.parse(timestr)
                if prev is None:
                    prev = t
                dt = t - prev 
                if dt.total_seconds() < HALFMIN:
                    continue
                
                sample.append(t)
                bid = row[2]
                if bid == "":
                    sample.append(None)
                else:
                    sample.append(float(bid))
                    
                ask = row[3]
                if ask == "":
                    sample.append(None)
                else:
                    sample.append(float(ask))
                    
                prev = t
                self.allTicks.append(sample)
        return
    
            