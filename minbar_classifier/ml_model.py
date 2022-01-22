from sklearn.naive_bayes import *
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import *
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pickle

class MLClassifier(object):
    def __init__(self):
        self.model = svm.SVC(C = 1., kernel='rbf')
        # model = GaussianNB()
        # model = MultinomialNB()
        # model = ComplementNB()
        # model = tree.DecisionTreeClassifier()
        # model = RandomForestClassifier()
        model = svm.SVC(C=1., kernel='rbf')
        # model = tf_nn.TFClassifier((x_train.shape[1],),3)
        # model = LogisticRegression(max_iter=1000)
        # model = XGBClassifier(use_label_encoder = False)
        # model = AdaBoostClassifier(n_estimators=300)

    def fit(self,x_train,y_train):
        self.model.fit(x_train,y_train)

    def predict(self,x_test):
        return self.model.predict(x_test)

    def save(self,mf):
        pickle.dump(self.model, open(mf, 'wb'))