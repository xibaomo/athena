'''
Created on Sep 4, 2018

@author: fxua
'''
from apps.spm.spamfilter import SpamFilter
from apps.spm.spam_multifilters import SpamMultiFilters
from apps.fts.forex_tick_sampler import ForexTickSampler
from apps.generic_app.generic_app import GenericApp
from apps.forex_trainer.fx_trainer_creator import createForexTrainer
from apps.gaopt.gaoptimizer import GaOptimizer
from apps.forex_bar_trainer.forex_bar_trainer import ForexBarTrainer
AppSwitcher = {
    99: SpamFilter.getInstance,
    98: SpamMultiFilters.getInstance,
    
    0:  ForexTickSampler.getInstance,
    1:  createForexTrainer,
    2:  GenericApp.getInstance,
    3:  GaOptimizer.getInstance,
    4:  ForexBarTrainer.getInstance
    }

def createApp(appType):
    func = AppSwitcher.get(appType,"No such an app")
    return func()