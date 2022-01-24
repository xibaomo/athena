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

        hi_rtn = df[HIGH_KEY]/df[OPEN_KEY] - 1.
        lw_rtn = df[LOW_KEY]/df[OPEN_KEY]-1.
        fm = []
        for tid in time_id:
            past_op = df[OPEN_KEY][tid - lookback:tid].values
            rtn_op = np.diff(np.log(past_op))
            rtn_hi = hi_rtn.values[tid-lookback+1:tid]
            rtn_lw = lw_rtn.values[tid-lookback+1:tid]
            rtn = np.vstack((rtn_op,rtn_hi,rtn_lw))
            rtn = np.transpose(rtn)

            fm.append(rtn)

        fm = np.array(fm)

        return fm,time_id,hour_lookback

