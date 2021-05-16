#include "multi_pairtrader.h"
#include "pyrunner/pyrunner.h"
#include "linreg/roblinreg.h"
using namespace std;
using namespace athena;

Message
MultiPairTrader::processMsg(Message& msg)
{
    Message outmsg;

    FXAct action = (FXAct)msg.getAction();
    switch(action) {
    case FXAct::CHECKIN:
        outmsg = procMsg_noreply(msg,[this](Message& msg) {
            Log(LOG_INFO) << "Client checked in";
        });
        break;
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
MultiPairTrader::procMsg_ASK_PAIR(Message& msg)
{
    selectTopCorr();

    Message out;
    out.setAction(FXAct::ASK_PAIR);
    return out;
}
Message
MultiPairTrader::procMsg_SYM_HIST_OPEN(Message& msg)
{
    char* pc = (char*)msg.getChar();
    size_t cb = msg.getCharBytes() - 2*sizeof(int);
    String sym = String(pc + 2*sizeof(int),cb);
    int len = msg.getDataBytes()/sizeof(real32);
    Log(LOG_INFO) << "Received history: " + sym;
    Log(LOG_INFO) << "History length: " + to_string(len);

    if (m_sym2hist.find(sym) != m_sym2hist.end()) {
        Log(LOG_FATAL) << "Duplicated symbol received: " + sym;
    }


    real32* pm = (real32*)msg.getData();
    std::vector<real32> v(pm,pm+len);
    m_sym2hist[sym] = std::move(v);

    Message out;
    return out;
}

void
MultiPairTrader::selectTopCorr()
{
    Log(LOG_INFO) << "Total symbols received: " + to_string(m_sym2hist.size());
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
                Log(LOG_INFO) << "Testing cointegration: " +  keys[i] + " vs " + keys[j];
                if (!test_coint(v1,v2)) {
                    continue;
                }

                SymPair sp{keys[i],keys[j],corr};
                m_topCorrSyms.push_back(sp);

                RobLRParam pm = robLinreg(v1,v2);
                if (pm.c0 > 0.)
                    Log(LOG_INFO) << "Top corr pair: " + keys[i] + " , " + keys[j] + ": " +to_string(corr);
                else
                    Log(LOG_INFO) << "Top corr pair: " + keys[j] + " , " + keys[i] + ": " +to_string(corr);

                Log(LOG_INFO) << "R2 = " + to_string(pm.r2);
            }
        }
    }

    Log(LOG_INFO) << "Inspected pairs: " + to_string(k);
}
