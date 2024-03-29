from enum import IntEnum

DATE_KEY = '<DATE>'
TIME_KEY = '<TIME>'
OPEN_KEY = '<OPEN>'
HIGH_KEY = '<HIGH>'
LOW_KEY  = '<LOW>'
CLOSE_KEY = '<CLOSE>'
TICKVOL_KEY = '<TICKVOL>'
SPREAD_KEY = '<SPREAD>'
STD_KEY  = 'STD'
MID_KEY = 'MID'
RET_KEY = "RET"
ENDDATE_KEY = "END_DATETIME"
TIMESTAMP_KEY = "TIMESTAMP"

class ModelType(IntEnum):
    ML = 0,
    DNN = 1

class FeatureType(IntEnum):
    PREDEFINED = 0,
    RAW = 1