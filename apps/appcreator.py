'''
Created on Sep 4, 2018

@author: fxua
'''
from apps.spm.spamfilter import SpamFilter

AppSwitcher = {
    99: SpamFilter.getInstance
    }

def createApp(appType):
    func = AppSwitcher[appType]
    return func()