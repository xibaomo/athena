/*
 * =====================================================================================
 *
 *       Filename:  ma_hunter.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/18/2019 03:18:00 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "ma_grad_hunter.h"
#include <fstream>
using namespace std;
using namespace athena;

MAHunter::MAHunter(const String& cfg,MACalculator* cal) : MABasePredictor(cal),m_config(nullptr)
{
    m_config = &MahuntConfig::getInstance();
    m_config->loadConfig(cfg);
}

MAHunter::~MAHunter()
{
    dumpRecords();
}

void
MAHunter::dumpRecords()
{
    ofstream ofs("ma.csv");
    ofs << "MA"<<endl;
    for (auto& p : m_records) {
        ofs << p << endl;
    }
    ofs.close();
}
void
MAHunter::prepare()
{
    // Compute median
    m_median.resize(m_allMinBars->size());
    for(size_t i = 0; i < m_allMinBars->size(); i++) {
        auto& mb = (*m_allMinBars)[i];
        m_median[i] = (mb.high + mb.low) *.5;
    }

    m_maCal->compAllMA(m_median,m_config->getMALookback(),m_ma);
}

FXAction
MAHunter::predict()
{
    auto& mb = m_allMinBars->back();
    real32 md = (mb.high + mb.low)*.5;
    m_median.push_back(md);

    real32 ma = m_maCal->compLatestMA(m_median,m_config->getMALookback(),m_median.size()-1);
    m_ma.push_back(ma);

    m_records.push_back(ma);

    vector<real32> ma_aux(m_ma.begin()+10000,m_ma.end());
    vector<real32> curve;
    savgol_smooth1D(ma_aux,51,3,curve);

    int tp = findNearestTurnPoint(curve);
    int id = ma_aux.size() - 2;
    int offset = id - tp;
    if (offset > m_config->getTurnPointOffset())
        return FXAction::NOACTION;

    real32 slope = (ma_aux[id+1]-ma_aux[id-1])/2.;
    real32 fos    = m_config->getFireOffSlope();
    if (slope >= fos)
        return FXAction::PLACE_BUY;

    if (slope <= -fos)
        return FXAction::PLACE_SELL;

    return FXAction::NOACTION;
}

int
MAHunter::findNearestTurnPoint(vector<real32>& curve)
{
    int id = curve.size() - 1;
    while(id>0) {
        real32 c = curve[id];
        if (c > curve[id+1] && c > curve[id-1]) {
            return id;
        } else if(c < curve[id+1] && c < curve[id-1]) {
            return id;
        } else {
            id--;
        }
    }

    return id;
}
//int
//MAHunter::findNearestTurnPoint()
//{
//    int id =  m_ma.size()-2;
//    while(id > 0) {
//        real32 slope = (m_ma[id+1] - m_ma[id-1])/2.;
//        if (fabs(slope) < m_config->getSlopeZero())
//            return id;
//        id--;
//    }
//}
