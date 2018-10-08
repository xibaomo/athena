'''
Created on Sep 4, 2018

@author: fxua
'''
from apps.spm.spamfilter import SpamFilter
from apps.spm.spam_multifilters import SpamMultiFilters
from apps.fts.forex_tick_sampler import ForexTickSampler
from apps.forex_trainer.forex_mfilters import ForexMultiFilters
AppSwitcher = {
    99: SpamFilter.getInstance,
    98: SpamMultiFilters.getInstance,
    
    0:  ForexTickSampler.getInstance,
    1:  ForexMultiFilters.getInstance
    }

def createApp(appType):
    func = AppSwitcher.get(appType,"No such an app")
    return func()