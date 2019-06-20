#include "multi_pairtrader.h"
#include "pyrunner/pyrunner.h"
using namespace std;
using namespace athena;

Message
MultiPairTrader::processMsg(Message& msg)
{
    Message outmsg;

    FXAction action = (FXAction)msg.getAction();
    switch(action) {
    case FXAction::CHECKIN:
        outmsg = procMsg_noreply(msg,[this](Message& msg) {
            Log(LOG_INFO) << "Client checked in";
        });
        break;
    case FXAction::SYM_HIST_OPEN:
        outmsg = procMsg_SYM_HIST_OPEN(msg);
        break;
    case FXAction::ASK_PAIR:
        outmsg = procMsg_ASK_PAIR(msg);
        break;
    default:
        break;
    }

    return outmsg;
}

Message
MultiPairTrader::procMsg_ASK_PAIR(Message& msg)
{
    selectTopCorr();
}
Message
MultiPairTrader::procMsg_SYM_HIST_OPEN(Message& msg)
{
    char* pc = (char*)msg.getChar();
    size_t cb = msg.getCharBytes() - 2*sizeof(int);
    String sym = String(pc + 2*sizeof(int),cb);

    Log(LOG_INFO) << "Received history: " + sym;

    if (m_sym2hist.find(sym) != m_sym2hist.end()) {
        Log(LOG_FATAL) << "Duplicated symbol received: " + sym;
    }

    int len = msg.getDataBytes()/sizeof(real32);
    real32* pm = (real32*)msg.getData();
    std::vector<real32> v(pm,pm+len);
    m_sym2hist[sym] = std::move(v);

    Message out;
    return out;
}

//bool
//MultiPairTrader::test_coint(std::vector<real32>& v1, std::vector<real32>& v2)
//{
//    CPyObject lx = PyList_New(v1.size());
//    CPyObject ly = PyList_New(v2.size());
//    for (size_t i = 0; i < v1.size(); i++) {
//        PyList_SetItem(lx,i,Py_BuildValue("f",v1[i]));
//        PyList_SetItem(ly,i,Py_BuildValue("f",v2[i]));
//    }
//
//    CPyObject args = Py_BuildValue("(OO)",lx.getObject(),ly.getObject());
//    PyRunner& pyrun = PyRunner::getInstance();
//
//    CPyObject res = pyrun.runAthenaPyFunc("coint","coint_verify",args);
//
//    if (PyInt_AsLong(res) == 1) {
//        return true;
//    }
//
//    return false;
//
//}

void
MultiPairTrader::selectTopCorr()
{
    vector<String> keys;
    for(const auto& kv : m_sym2hist) {
        keys.push_back(kv.first);
    }

    for (size_t i = 0; i < keys.size(); i++) {
        for(size_t j=i+1; j < keys.size(); j++) {
            auto& v1 = m_sym2hist[keys[i]];
            auto& v2 = m_sym2hist[keys[j]];
            auto corr = computePairCorr(v1,v2);
            if (fabs(corr) > m_cfg->getCorrBaseline()) {
                Log(LOG_INFO) << "Testing cointegration: " +  keys[i] + " vs " + keys[j];
                if (!test_coint(v1,v2)) {
                    continue;
                }

                SymPair sp{keys[i],keys[j],corr};
                m_topCorrSyms.push_back(sp);
                if (v1[0] < v2[0])
                    Log(LOG_INFO) << "Top coor pair: " + keys[i] + "," + keys[j] + ": " +to_string(corr);
                else
                    Log(LOG_INFO) << "Top coor pair: " + keys[j] + "," + keys[i] + ": " +to_string(corr);
            }
        }
    }
}
