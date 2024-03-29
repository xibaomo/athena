#include "pair_selector.h"
#include "pyrunner/pyrunner.h"
#include "linreg/roblinreg.h"
using namespace std;
using namespace athena;

Message
PairSelector::processMsg(Message& msg) {
    Message outmsg;

    FXAct action = (FXAct)msg.getAction();
    switch(action) {
    case FXAct::SYM_HIST_OPEN:
        outmsg = procMsg_SYM_HIST_OPEN(msg);
        break;
    case FXAct::ASK_PAIR:
        outmsg = procMsg_ASK_PAIR(msg);
        break;
    default:
        break;
    }

    return outmsg;
}

Message
PairSelector::procMsg_ASK_PAIR(Message& msg) {
    selectTopCorr();

    Message out;
    out.setAction(FXAct::ASK_PAIR);
    return out;
}
Message
PairSelector::procMsg_SYM_HIST_OPEN(Message& msg) {
    SerializePack pack;
    unserialize(msg.getComment(),pack);
    String sym = pack.str_vec[0].substr(0,6);

    Log(LOG_INFO) << "Received history: " + sym <<std::endl;
    Log(LOG_INFO) << "History length: " + to_string(pack.real64_vec.size()) <<std::endl;

    if (m_sym2hist.find(sym) != m_sym2hist.end()) {
        Log(LOG_ERROR) << "Duplicated symbol received: " + sym <<std::endl;
    }

    m_sym2hist[sym] = std::move(pack.real64_vec);
    Message out;
    return out;
}

void
PairSelector::selectTopCorr() {
    Log(LOG_INFO) << "Total symbols received: " + to_string(m_sym2hist.size()) <<std::endl;
    vector<String> keys;
    for(const auto& kv : m_sym2hist) {
        keys.push_back(kv.first);
    }

    int k = 0;

    for (size_t i = 0; i < keys.size(); i++) {
        for(size_t j=i+1; j < keys.size(); j++) {
            auto& v1 = m_sym2hist[keys[i]];
            auto& v2 = m_sym2hist[keys[j]];

            k++;
//            String fn = keys[i]+"_"+keys[j]+".csv";
//            dump_csv(fn,v1,v2);

            auto corr = computePairCorr(v1,v2);
            if (fabs(corr) > m_cfg->getCorrBaseline()) {
                std::cout<<std::endl;
                Log(LOG_INFO) << "Testing cointegration: " +  keys[i] + " vs " + keys[j] <<std::endl;
                real64 pv = m_cfg->getCoIntPVal();
                if (!test_coint(v1,v2,pv)) {
                    continue;
                }

                SymPair sp{keys[i],keys[j],corr};
                m_topCorrSyms.push_back(sp);

                LRParam pm = ordLinreg(v1,v2);
                if (pm.c0 > 0.)
                    Log(LOG_INFO) << "Top corr pair: " + keys[i] + " , " + keys[j] + ": " +to_string(corr) <<std::endl;
                else
                    Log(LOG_INFO) << "Top corr pair: " + keys[j] + " , " + keys[i] + ": " +to_string(corr) <<std::endl;

                Log(LOG_INFO) << "R2 = " + to_string(pm.r2) <<std::endl;
            }
        }
    }

    Log(LOG_INFO) << "Inspected pairs: " + to_string(k) <<std::endl;
}

void PairSelector::finish() {
    PyObject* mod = PyImport_ImportModule("pd_utils");
    if (!mod)
        Log(LOG_FATAL) << "Failed to import module: " <<std::endl;

    PyObject* func = PyObject_GetAttrString(mod,"addCol");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: " <<std::endl;

    int k=0;
    for (auto& it : m_sym2hist) {
            k++;
        auto& v = it.second;
        PyObject* lx = PyList_New(v.size());

        for (size_t i = 0; i < v.size(); i++) {
            PyList_SetItem(lx,i,Py_BuildValue("d",v[i]));
        }

        PyObject* header = Py_BuildValue("s",it.first.c_str());
        PyObject* args = Py_BuildValue("(OO)",header,lx);
        PyObject_CallObject(func,args);

        Py_DECREF(args);
        Py_DECREF(lx);
        Py_DECREF(header);
    }

    Log(LOG_INFO) << "All syms added to pd dataframe: " << k <<endl;
    PyObject* str = Py_BuildValue("s","sym_hist.csv");
    PyObject* ag = Py_BuildValue("(O)",str);
    PyRunner::getInstance().runAthenaPyFunc("pd_utils","dump_csv",ag);

    Py_DECREF(str);

}
