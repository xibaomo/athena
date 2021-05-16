/*
 * =====================================================================================
 *
 *       Filename:  ma_crosser.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  05/12/2019 05:12:53 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "ma_crosser.h"
#include "mcconf.h"
#include <fstream>
using namespace std;
using namespace athena;

MACrosser::MACrosser(const String& cf, MACalculator* cal): MABasePredictor(cal), m_config(nullptr) {
    m_config = &MACrosserConfig::getInstance();
    m_config->loadConfig(cf);

    Log(LOG_INFO) << "MA crosser created";
}

MACrosser::~MACrosser()
{
    dumpRecords();
}

void
MACrosser::dumpRecords()
{
    ofstream ofs("action.csv");
    ofs<<"LMA,SMA,ACTION"<<endl;
    for (auto& ac : m_records) {
        ofs << ac.long_ma << ","
            << ac.short_ma<<","
            << (int) ac.action<<endl;
    }
    ofs.close();
}
void
MACrosser::prepare()
{
        // Compute median
    m_median.resize(m_allMinBars->size());
    for(size_t i = 0; i < m_allMinBars->size(); i++) {
        auto& mb = (*m_allMinBars)[i];
        m_median[i] = (mb.high + mb.low) *.5;
    }

    m_maCal->compAllMA(m_median,m_config->getLongLookBack(),m_long_ma);
    m_maCal->compAllMA(m_median,m_config->getShortLookBack(),m_short_ma);
}

FXAct
MACrosser::predict()
{
    auto& mb = m_allMinBars->back();
    real32 md = (mb.high + mb.low) *.5;
    m_median.push_back(md);

    real32 s_ma = m_maCal->compLatestMA(m_median,m_config->getShortLookBack(),
                                        m_median.size()-1);
    real32 l_ma = m_maCal->compLatestMA(m_median,m_config->getLongLookBack(),
                                        m_median.size()-1);
    m_long_ma.push_back(l_ma);
    m_short_ma.push_back(s_ma);

    int tp = findNearestCross();
    int cid = m_long_ma.size()-1;

    FXAct action = FXAct::NOACTION;

    if (cid - tp > m_config->getTurnPointExpiry()) {
        action = FXAct::NOACTION;
        m_records.emplace_back(l_ma,s_ma,action);
        return action;
    }

    real32 gap = m_long_ma.back() - m_short_ma.back();
    if ( fabs(gap) < m_config->getStartOffGap()) {
        action = FXAct::NOACTION;
        m_records.emplace_back(l_ma,s_ma,action);
        return action;
    }

    if (gap > 0) {
//        action =  FXAct::PLACE_SELL;
        action =  FXAct::PLACE_BUY;
    } else {
//        action =  FXAct::PLACE_BUY;
        action =  FXAct::PLACE_SELL;
    }

    m_records.emplace_back(l_ma,s_ma,action);
    return action;
}

int
MACrosser::findNearestCross()
{
    real32 cs = m_long_ma.back() - m_short_ma.back(); //current status
    int id = m_long_ma.size()-2;

    while (id > 0) {
        real32 ps = m_long_ma[id] - m_short_ma[id];
        if (ps * cs < 0)
            break;
        id--;
    }

    return id;
}
