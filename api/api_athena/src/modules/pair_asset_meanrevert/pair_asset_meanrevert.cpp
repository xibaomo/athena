/*
 * =====================================================================================
 *
 *       Filename:  pair_asset_meanrevert.cpp
 *
 *    Description:  the paired symbols satisfy y = a*x + b + epsilon(x), equivalently, we are trading
 *                  a custom symbol (y - a*x), which is stationary.
 *                  In most cases, when lot size of y and/or x is quite small, the ratio of y and x hardly
 *                  ensures (y-a*x) is stationary.
 *                  This concrete oracle, will carefully select lot sizes of both x and y, so as to ensure
 *                  (lot_y*y +/- lot_x*x) is stationary, and the rest is still classical mean-reverting algorithm.
 *
 *
 *        Version:  1.0
 *        Created:  07/15/2021 03:56:03 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "pair_asset_meanrevert.h"
#include "basics/utils.h"
#include "linreg/linreg.h"
#include <numeric>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_statistics.h>
#define STATIONARY_PV 0.02
#define CORR_LIMIT 0.85
using namespace std;
using namespace athena;

PairAssetMeanRevert::~PairAssetMeanRevert() {
    Log(LOG_INFO) << "Num of buys: " + to_string(m_buys) + ", sells: " + to_string(m_sells);

    dumpVectors("buysell_totalassets.csv",m_totalAssets_buy,m_totalAssets_sell);
    dumpVectors("all_totalassets.csv",m_totalAssets_mid);

    ostringstream os;
    os << "Dev range: buy: [" << m_lowBuyDev<<","<<m_highBuyDev<<"], sell: [" << m_lowSellDev << "," << m_highSellDev << "]";
    Log(LOG_INFO) << os.str();

}
#if 0
static real64 ratio_cost(const gsl_vector* v, void* params) {
    real64 x = gsl_vector_get(v,0);
    real64 y = gsl_vector_get(v,1);
    real64* p = (real64*)params;

    real64 c = (y/x - p[0]);
    return c*c;
}

static
void ratio_cost_df(const gsl_vector* v, void* params, gsl_vector* df) {
    real64 x = gsl_vector_get(v,0);
    real64 y = gsl_vector_get(v,1);
    real64* p = (real64*)params;

    real64 gx = 2.f*(y/x-p[0])*(-y/x/x);
    real64 gy = 2.f*(y/x-p[0])*1.f/x;
    gsl_vector_set(df,0,gx);
    gsl_vector_set(df,1,gy);
}

static
void ratio_cost_fdf(const gsl_vector* v, void* params, real64* f, gsl_vector* df) {
    *f = ratio_cost(v,params);
    ratio_cost_df(v,params,df);
}
static
std::tuple<real64,real64> minimize_ratio_cost(real64 x0, real64 y0, real64 target) {
    gsl_vector* x = gsl_vector_alloc(2);
    gsl_multimin_function_fdf func;
    func.n = 2;
    func.f = ratio_cost;
    func.df = ratio_cost_df;
    func.fdf = ratio_cost_fdf;
    func.params = &target;

    gsl_vector_set(x,0,x0);
    gsl_vector_set(x,1,y0);

    int status,iter(0);
    const gsl_multimin_fdfminimizer_type* T = gsl_multimin_fdfminimizer_conjugate_fr;
    gsl_multimin_fdfminimizer* s = gsl_multimin_fdfminimizer_alloc(T,2);
    gsl_multimin_fdfminimizer_set(s,&func,x,0.01,1e-4);

    do {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate(s);
        if(status) break;

        status = gsl_multimin_test_gradient(s->gradient,1e-3);
        if(status==GSL_SUCCESS) {
            ostringstream oss;
            oss << "Minimum found: " << s->f << " at (" << gsl_vector_get(s->x,0) << "," << gsl_vector_get(s->x,1)<<")";
            Log(LOG_INFO) << oss.str();
        }
    } while(status==GSL_CONTINUE && iter <500);

    real64 xp = gsl_vector_get(s->x,0);
    real64 yp = gsl_vector_get(s->x,1);
    xp = floor(xp*100.f)/100.f;
    yp = floor(yp*100.f)/100.f;

    ostringstream oss;
    oss << "lot_x: " << xp << ", lot_y: " << yp << ", lot_x/log_y = " << xp/yp;
    Log(LOG_INFO) << oss.str();
    gsl_vector_free(x);
    gsl_multimin_fdfminimizer_free(s);

    return std::make_tuple(xp,yp);
}
#endif // 0
void
PairAssetMeanRevert::findBestLots(real64 lotx0, real64 loty0) {
    auto& asset_x = m_trader->getAssetX_mid();
    auto& asset_y = m_trader->getAssetY_mid();

    LRParam param = linreg(&asset_x[0],&asset_y[0],asset_x.size());

    Log(LOG_INFO) << "Linear reg: c0: " + to_string(param.c0) + ", c1: " + to_string(param.c1);

    //auto pm = minimize_ratio_cost(lotx0,loty0,1.f/abs(param.c1));

    m_loty = 0.05;
    m_lotx = abs(m_loty*param.c1);

    if (param.c1 > 0)
        m_pairPosDir = OPPOSITE;
    else
        m_pairPosDir = SAME;
}
void
PairAssetMeanRevert::init() {
    findBestLots(0.05f,0.05f);

    createTotalAssets();

    real64 pv = testADF(&m_totalAssets_mid[0],m_totalAssets_mid.size());
    Log(LOG_INFO) << "Total asset stationary p-value: " + to_string(pv);

    if(pv < STATIONARY_PV) m_stationaryState = YES;

    m_curMean = std::accumulate(m_totalAssets_mid.begin(),m_totalAssets_mid.end(),0.f) / (real64)m_totalAssets_mid.size();
    Log(LOG_INFO) << "Initial mean of total assets: " + to_string(m_curMean);

}
void
PairAssetMeanRevert::createTotalAssets() {
    auto& asx = m_trader->getAssetX_mid();
    auto& asy = m_trader->getAssetY_mid();

    m_totalAssets_mid.resize(asx.size());
    for(size_t i=0; i < asx.size(); i++) {
        m_totalAssets_mid[i] = m_lotx * asx[i] + m_loty * asy[i];
    }
}

real64
PairAssetMeanRevert::compDevFromMean() {
    int std_len = m_lookback;
    int start = m_totalAssets_mid.size() - std_len;
    m_devUnit = gsl_stats_sd(&m_totalAssets_mid[start],1,std_len);

    Log(LOG_INFO) << "std of past " + to_string(std_len) + ": " + to_string(m_devUnit);

    m_curMean = std::accumulate(m_totalAssets_mid.begin()+start,m_totalAssets_mid.end(),0.f) / (real64)m_lookback;
    Log(LOG_INFO) << "Mean updated: " + to_string(m_curMean);

    DevInfo dev;
    dev.buy = (m_totalAssets_buy.back() - m_curMean) / m_devUnit;
    dev.sell = (m_totalAssets_sell.back() - m_curMean) / m_devUnit;

    m_devs.push_back(dev);

    if (dev.buy > m_highBuyDev) m_highBuyDev = dev.buy;
    if (dev.buy < m_lowBuyDev)  m_lowBuyDev = dev.buy;
    if (dev.sell > m_highSellDev) m_highSellDev = dev.sell;
    if (dev.sell < m_lowSellDev)  m_lowSellDev = dev.sell;

    Log(LOG_INFO) << "Deviation from mean (std unit), buy: " + to_string(dev.buy) + ", sell: " + to_string(dev.sell);
}

void
PairAssetMeanRevert::appendAssets() {
    auto& ax_buy = m_trader->getAssetX_buy();
    auto& ay_buy = m_trader->getAssetY_buy();
    auto& ax_sell = m_trader->getAssetX_sell();
    auto& ay_sell = m_trader->getAssetY_sell();
    switch(m_pairPosDir) {
    case PairPosDirection::SAME: {
        real64 total_asset_buy = m_loty*ay_buy.back() + m_lotx * ax_buy.back();
        real64 total_asset_sell = m_loty*ay_sell.back() + m_lotx * ax_sell.back();
        m_totalAssets_buy.push_back(total_asset_buy);
        m_totalAssets_sell.push_back(total_asset_sell);
        real64 mid = (total_asset_buy+total_asset_sell)*.5f;
        m_totalAssets_mid.push_back(mid);
    }
    break;
    case PairPosDirection::OPPOSITE: {
        real64 total_asset_buy = m_loty * ay_buy.back() - m_lotx*ax_sell.back();
        real64 total_asset_sell= m_loty * ay_sell.back() - m_lotx*ax_buy.back();
        m_totalAssets_buy.push_back(total_asset_buy);
        m_totalAssets_sell.push_back((total_asset_sell));
        real64 mid = (total_asset_buy+total_asset_sell)*.5f;
        m_totalAssets_mid.push_back(mid);
    }
    break;
    default:
        break;
    }
}

void
PairAssetMeanRevert::checkStationaryState() {
    int start = m_totalAssets_mid.size() - m_lookback;
    real64 pv = testADF(&m_totalAssets_mid[start],m_lookback);
    Log(LOG_INFO) << "Past " + to_string(m_lookback) + " total asset stationary p-value: " + to_string(pv);
    if (pv > STATIONARY_PV)
        m_stationaryState = NO;
    else
        m_stationaryState = YES;
}

FXAct
PairAssetMeanRevert::decision_stationary() {
    compDevFromMean();

    MptConfig* cfg = m_trader->getConfig();

    real64 low_thd = cfg->getLowThresholdStd();
    real64 high_thd = cfg->getHighThresholdStd();

    auto& dev = m_devs.back();
    if (dev.buy <= -high_thd) {
        m_buys++;
        return FXAct::PLACE_BUY;
    }

    if (dev.sell >= high_thd) {
        m_sells++;
        return FXAct::PLACE_SELL;
    }

    if ( abs(dev.sell) <= low_thd ) {
        return FXAct::CLOSE_BUY;
    }

    if ( abs(dev.buy) <= low_thd ) {
        return FXAct::CLOSE_SELL;
    }

    real64 old_dev = m_devs[m_devs.size()-2].sell;
    real64 new_dev = m_devs.back().sell;
    if ( old_dev * new_dev < 0 ) {
        return FXAct::CLOSE_BUY;
    }

    old_dev = m_devs[m_devs.size()-2].buy;
    new_dev = m_devs.back().buy;
    if ( old_dev * new_dev < 0 ) {
        return FXAct::CLOSE_SELL;
    }
    return FXAct::NOACTION;
}

FXAct
PairAssetMeanRevert::decision_non_stationary() {
    int start = m_totalAssets_mid.size() - m_lookback;
    std::vector<real64> x(m_lookback);
    std::iota(x.begin(),x.end(),0.f);

    real64 corr = gsl_stats_correlation(&x[0],1,&m_totalAssets_mid[start],1,m_lookback);

    Log(LOG_INFO) << "Corr of total assets vs time: " + to_string(corr);

    if (corr >= CORR_LIMIT) {
        m_buys++;
        return FXAct::PLACE_BUY;
    }
    if (corr <= -CORR_LIMIT) {
        m_sells++;
        return FXAct::PLACE_SELL;
    }

    return FXAct::NOACTION;
}
FXAct
PairAssetMeanRevert::getDecision() {
    appendAssets();

    checkStationaryState();

    switch(m_stationaryState) {
    case StationaryState::YES:
        return decision_stationary();

    case StationaryState::NO:
        return decision_non_stationary();

    default:
        break;
    }

    return FXAct::NOACTION;
}

bool
PairAssetMeanRevert::isContinue() {
    return true;
}
