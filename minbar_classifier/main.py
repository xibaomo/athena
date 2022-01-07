# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys, os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import *
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import *
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import tf_nn
from labeling import *
from features import *
from logger import *
from tf_nn import *
from prediction import *
from basics import *
from conf import *
import pickle

def loadcsv(fn):
    df = pd.read_csv(fn, sep='\t')
    ret = np.diff(np.log(df[OPEN_KEY].values))
    ret = np.append(ret,0)
    df[RET_KEY] = ret
    return df

def train_model(x_train, y_train):
    Log(LOG_INFO) << "Training model..."
    #model = GaussianNB()
    # model = MultinomialNB()
    # model = ComplementNB()
    # model = tree.DecisionTreeClassifier()
    # model = RandomForestClassifier()
    # model = svm.SVC(C = 1., kernel='rbf')
    model = tf_nn.TFClassifier((1,x_train.shape[1]),3)
    # model = LogisticRegression(max_iter=1000)
    # model = XGBClassifier(use_label_encoder = False)
    # model = AdaBoostClassifier(n_estimators=300)
    ## fit the model
    # pdb.set_trace()
    model.fit(x_train, y_train)

    return model

def eval_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    Log(LOG_INFO) << "Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum())
    # calculate profit
    profit = 0.
    prof_buy=0
    prof_sell=0
    loss = 0.
    for i in range(len(y_test)):
        if y_pred[i] == Action.NO_ACTION:
            continue
        if y_pred[i] == y_test[i]:
            profit += 1
            if y_pred[i] == Action.BUY:
                prof_buy+=1
            if y_pred[i] == Action.SELL:
                prof_sell+=1
        else:
            loss += 1

    Log(LOG_INFO)<<"true deals: buy: %d, sell: %d"%(( (y_test==Action.BUY).sum(),(y_test==Action.SELL).sum()))
    Log(LOG_INFO) << "predicted deals: buy: %d, sell: %d" % (((y_pred == Action.BUY).sum(), (y_pred == Action.SELL).sum()))

    Log(LOG_INFO) << "profit = %d, loss=%d, net=%d" % (profit, loss, profit - loss)
    Log(LOG_INFO) << "profit buy: %d, sell: %d" % (prof_buy,prof_sell)
    Log(LOG_INFO) << "win ratio: %.3f" % (profit / (profit + loss))

    #### blind trade: buy,sell,buy,sell,...
    blind_trade = np.ones(len(y_test))
    for i in range(len(blind_trade)):
        # if i % 2==1:
        if np.random.random() > 0.5:
            blind_trade[i] = Action.BUY
        else:
            blind_trade[i] = Action.SELL
    Log(LOG_INFO) << "blind win ratio: %.3f" % ((y_test==blind_trade).sum()*1./len(y_test))
    Log(LOG_INFO) << "all long win ratio: %.3f" % ((y_test == Action.BUY).sum() * 1. / len(y_test))

    # Log(LOG_INFO) << "estimating feature importances ..."
    # perm_importance = permutation_importance(model, x_test, y_test)
    # Log(LOG_INFO) << "importance: "+ str(perm_importance.importances_mean)

def dumpTestSet(df,used_time_id,labels, end_time, end_high,end_low,test_size):
    dff = pd.DataFrame()
    tid_s = len(used_time_id) - test_size
    for i in range(tid_s,len(used_time_id)):
        tid = used_time_id[i]
        dff = dff.append(df.loc[tid,[DATE_KEY,TIME_KEY,OPEN_KEY,SPREAD_KEY]])
    dff.reset_index(drop=True)
    label_aux = labels[tid_s:].astype(int)
    for i in range(len(label_aux)):
        if label_aux[i] == Action.BUY:
            label_aux[i] = 1
            continue
        if label_aux[i] == Action.SELL:
            label_aux[i] = -1
            continue
    dff['LABEL'] = label_aux
    dff['END_TIME'] = end_time[tid_s:]
    dff['END_HIGH'] = end_high[tid_s:]
    dff['END_LOW']  = end_low[tid_s:]

    dff.to_csv("test_set.csv",index=False)

def dumpTestFeatures(df,time_id_test,fm_test):
    dff = pd.DataFrame()
    dff[DATE_KEY] = df[DATE_KEY][time_id_test]
    dff[TIME_KEY] = df[TIME_KEY][time_id_test]
    for i in range(fm_test.shape[1]):
        header = "F_%d" % i
        dff[header] = fm_test[:,i]

    dff.to_csv("offline_feature_testset.csv",index=False)

if __name__ == '__main__':
    Log.setlogLevel(LOG_INFO)

    if len(sys.argv) < 3:
        Log(LOG_FATAL) << "Usage: " + sys.argv[0] + "<sym>.csv " + "<config>.yaml"
    fn = sys.argv[1]
    cf = sys.argv[2]
    df = loadcsv(fn)

    config = MasterConf(cf)

    timestamp = pd.to_datetime(df[DATE_KEY] + " " + df[TIME_KEY])
    dt = (timestamp[1] - timestamp[0])
    dtmin = dt.seconds/60
    Log(LOG_INFO) << "Minbar spacing is %d min" % dtmin
    labels,time_id,end_time,end_high,end_low = later_change_label(df,config.getReturnThreshold(),config.getTrueReturnRatio(),
                                                                  config.getPosLifeSec(),int(dtmin))
    test_size = config.getTestSize()
    Log(LOG_INFO) << "Test size: %d" % test_size
    fexconf = FexConfig(cf)
    fm,used_time_id,lookback = prepare_features(fexconf, df, time_id)
    Log(LOG_INFO) << "Feature dimension: %d" % fm.shape[1]
    used_labels = labels[lookback:]
    used_endtime = end_time[lookback:]
    tid_s = used_time_id[-test_size]
    tid_e = used_time_id[-1]
    Log(LOG_INFO) << "start date of test: " + df[DATE_KEY][tid_s] + " " + df[TIME_KEY][tid_s]
    Log(LOG_INFO) << "end   date of test: " + df[DATE_KEY][tid_e] + " " + df[TIME_KEY][tid_e]
    dumpTestSet(df,used_time_id,used_labels,used_endtime,end_high[lookback:],end_low[lookback:],test_size)
    dumpTestFeatures(df,used_time_id[-test_size:],fm[-test_size:,:])
    x_train, y_train, x_test, y_test,scaler = split_dataset(fm,used_labels,test_size)

    model = train_model(x_train, y_train)
    pickle.dump(model,open(config.getModelFile(),'wb'))
    pickle.dump(scaler,open(config.getScalerFile(),'wb'))

    if test_size == 0:
        sys.exit(0)

    Log(LOG_INFO) << "Evaluating model on training set..."
    eval_model(model,x_train,y_train)

    Log(LOG_INFO) << "Evaluating model on test set..."
    eval_model(model, x_test, y_test)
