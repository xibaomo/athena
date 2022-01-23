from basics import *
import numpy as np
from logger import *
class RawFex(object):
    def __init__(self,masterconf):
        self.yamlDict = masterconf.yamlDict
        Log(LOG_INFO) << "Raw fex created"

    def create_features(self,df,time_id):
        hour_lookback = 0
        lookback = self.yamlDict['RAW_FEATURES']['LOOKBACK']
        fm = []
        for tid in time_id:
            past_op = df[OPEN_KEY][tid-lookback:tid].values
            rtn = np.diff(np.log(past_op))
            fm.append(rtn)

        fm = np.array(fm)

        return fm,time_id,hour_lookback

