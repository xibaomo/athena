from apps.forex_trainer.forex_multifilters import ForexMultiFilters
from apps.forex_trainer.fxtconf import FxtConfig
from apps.forex_trainer.forex_dnn_trainer import ForexDNNTrainer

FxtCreator_switcher = {
    0: ForexMultiFilters.getInstance,
    1: ForexDNNTrainer.getInstance
    }
def createForexTrainer():
    config = FxtConfig()
    func = FxtCreator_switcher[config.getTrainerType()]
    
    return func()