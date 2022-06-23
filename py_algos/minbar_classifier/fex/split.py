from sklearn.preprocessing import *
from basics import *
from logger import *

def split_dataset_by_dates(df, fm, labels,time_id,start_time,end_time,config):
    idx = len(time_id)-1
    id_s = -1
    id_e = -1 # not included
    while idx>=0:
        tid = time_id[idx]
        t0 = df[TIMESTAMP_KEY][tid]
        dt = t0 - end_time
        #print(t0, end_time, dt.total_seconds())
        if dt.total_seconds() <= 0:
            id_e = idx+1
            break
        idx-=1
    while 1:
        tid = time_id[idx]
        t0 = df[TIMESTAMP_KEY][tid]
        dt = t0 - start_time
        if dt.total_seconds() <= 0:
            id_s = idx+1
            break
        idx -= 1

    test_size = id_e - id_s
    x_train = fm[:id_s,:]
    y_train = labels[:id_s]
    x_test  = fm[id_s:id_e]
    y_test  = labels[id_s:id_e]

    if len(y_test) == 0:
        Log(LOG_FATAL) << "Test dates cannot be found in history data"

    scaler = None
    if config.getFeatureType() == FeatureType.PREDEFINED:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    tid_s = time_id[id_s]
    tid_e = time_id[id_e-1]
    Log(LOG_INFO) << "train and test sets splitted"
    Log(LOG_INFO) << "Test size: %d" % test_size
    Log(LOG_INFO) << "start date of test: " + df[DATE_KEY][tid_s] + " " + df[TIME_KEY][tid_s]
    Log(LOG_INFO) << "end   date of test: " + df[DATE_KEY][tid_e] + " " + df[TIME_KEY][tid_e]

    return x_train,y_train,x_test,y_test, scaler

def split_dataset(fm, label, test_size):
    if test_size == 0:
        x_train = fm
        y_train = label
        x_test = np.array([])
        y_test = x_test
    else:
        x_train = fm[:-test_size, :]
        y_train = label[:-test_size]
        x_test = fm[-test_size:, :]
        y_test = label[-test_size:]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    Log(LOG_INFO) << "train and test sets splitted"

    return x_train, y_train, x_test, y_test, scaler