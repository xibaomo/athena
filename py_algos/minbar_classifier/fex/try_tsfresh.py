from main import loadcsv,train_model,eval_model
from conf import *
from labeling import *
from fex.features import *
from tsfresh import select_features

if __name__ == "__main__":
    Log.setlogLevel(LOG_INFO)

    if len(sys.argv) < 3:
        Log(LOG_FATAL) << "Usage: " + sys.argv[0] + "<sym>.csv " + "<config>.yaml"
    fn = sys.argv[1]
    cf = sys.argv[2]
    df = loadcsv(fn)

    config = MasterConf(cf)

    # labels,time_id = inst_change_label(df)
    labels, time_id = later_change_label(df, config.getReturnThreshold(), config.getPosLifeSec(),30)
    test_size = config.getTestSize()
    Log(LOG_INFO) << "Test size: %d" % test_size

    df_train = df.iloc[:-test_size,:]
    df_test = df.iloc[-test_size:,:]
    label_train = labels[:-test_size]
    label_test  = labels[-test_size:]

    fs = tsfresh(df,time_id,10)

    ff = select_features(fs,labels)

    Log(LOG_INFO) << "tsfresh gives %d features" % ff.shape[1]

    x_train,y_train,x_test,y_test,scaler = split_dataset(ff.values,labels,test_size)

    model = train_model(x_train, y_train)

    eval_model(model,x_test,y_test)