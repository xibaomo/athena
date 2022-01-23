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
            past_op = df[OPEN_KEY][tid - lookback:tid].values
            past_hi = df[HIGH_KEY][tid-lookback:tid].values
            past_lw = df[LOW_KEY][tid-lookback:tid].values
            past_cl = df[CLOSE_KEY][tid-lookback:tid].values
            rtn=[]
            for i in range(lookback):
                if past_op[i] > past_cl[i]:
                    rtn.append(past_lw[i]/past_hi[i]-1.)
                else:
                    rtn.append(past_hi[i]/past_lw[i]-1.)

            fm.append(rtn)

        fm = np.array(fm)

        return fm,time_id,hour_lookback

