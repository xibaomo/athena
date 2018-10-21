from apps.forex_trainer.forex_multifilters import ForexMultiFilters
from apps.forex_trainer.fxtconf import FxtConfig

FxtCreator_switcher = {
    0: ForexMultiFilters.getInstance,
    1: None
    }
def createForexTrainer():
    config = FxtConfig()
    func = FxtCreator_switcher[config.getTrainerType()]
    
    return func()