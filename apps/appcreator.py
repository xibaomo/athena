'''
Created on Sep 4, 2018

@author: fxua
'''
from apps.spm.spamfilter import SpamFilter
from apps.spm.spam_multifilters import SpamMultiFilters

AppSwitcher = {
    99: SpamFilter.getInstance,
    98: SpamMultiFilters.getInstance
    }

def createApp(appType):
    func = AppSwitcher.get(appType,"No such an app")
    return func()