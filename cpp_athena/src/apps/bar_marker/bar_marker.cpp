/*
 * =====================================================================================
 *
 *       Filename:  bar_marker.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/02/2018 03:35:14 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#include "bar_marker.h"
#include "parser.hpp"
#include "basics/log.h"
#include <vector>
using namespace std;
using namespace aria::csv;

BarMarker::BarMarker(const String& cfg) : App(cfg)
{
    m_pointValue["EURUSD"] = 1.E-5;
    m_pointValue["GBPUSD"] = 1.E-5;
    m_pointValue["USDCHF"] = 1.E-5;
    m_pointValue["USDJPY"] = 1.E-3;
    m_pointValue["USDCAD"] = 1.E-5;
    m_pointValue["AUDUSD"] = 1.E-5;

    m_symbol = getYamlValue("BAR_MARKER/FOREX_SYMBOL");
}
void
BarMarker::parseBarFile(const String& barFile)
{
    ifstream ifs(barFile);
    CsvParser parser = CsvParser(ifs).delimiter('\t');
    int k = -1;
    for (auto& row : parser) {
        k++;
        if (k==0) continue;
        vector<String> tmp;
        for(auto& field : row) {
            tmp.emplace_back(field);
        }

        MinBar mb;
        String timestr = tmp[0] + " " + tmp[1];
        TimePoint tp(boost::posix_time::time_from_string(timestr));
        mb.time = tp;
        mb.open = stof(tmp[2]);
        mb.high = stof(tmp[3]);
        mb.low  = stof(tmp[4]);
        mb.close= stof(tmp[5]);
        mb.tickvol = stoi(tmp[6]);

        m_allMinBars.emplace_back(mb);
    }
}

void
BarMarker::markMinBar()
{
    int offset_pts = stoi(getYamlValue("BAR_MARKER/OFFSET"));
    String sym = getYamlValue("BAR_MARKER/FOREX_SYMBOL");
    float pv = m_pointValue[sym];
    float offset = (float)offset_pts * pv;

    for (size_t i=0;i<m_allMinBars.size(); i++) {
        float pos= m_allMinBars[i].close;
        float ub = pos + offset;
        float lb = pos - offset;
        for (size_t j = i+1; j < m_allMinBars.size(); j++) {
            auto high = m_allMinBars[j].high;
            auto low  = m_allMinBars[j].low;

            if (high >= ub && low > lb ) {
                m_allMinBars[i].label = 0;
                break;
            } else if (low <= lb && ub > high) {
                m_allMinBars[i].label = 1;
                break;
            } else if (low <= lb && high >= ub) {
                m_allMinBars[i].label = 2;
                Log(LOG_WARNING) << "Rush minute: high = " + to_string(high) + ", low = " + to_string(low);
                break;
            } else {
                ;
            }
        }
    }
}

void
BarMarker::dumpCSV()
{
    const String csvfile = m_symbol + "_labeled.csv";
    ofstream ofs(csvfile);
    ofs << "TIME,OPEN,HIGH,LOW,CLOSE,TICKVOL,LABEL\n";

    for (auto& bar : m_allMinBars) {
        ofs << boost::posix_time::to_simple_string(bar.time) << ","
            << bar.open << ","
            << bar.high << ","
            << bar.low  << ","
            << bar.close << ","
            << bar.label << "\n";
    }

    ofs.close();
}
void
BarMarker::prepare()
{
    const String key = "BAR_MARKER/BAR_FILE";
    String barFile = getYamlValue(key);
    parseBarFile(barFile);
}

void
BarMarker::execute()
{
    markMinBar();
}

void
BarMarker::finish()
{
}
