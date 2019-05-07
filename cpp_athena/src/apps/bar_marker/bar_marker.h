/*
 * =====================================================================================
 *
 *       Filename:  bar_marker.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/02/2018 03:31:16 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _BAR_MARKER_H_
#define  _BAR_MARKER_H_

#include "app.h"
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <unordered_map>

typedef boost::posix_time::ptime TimePoint;
struct MinBar {
    String timestr;
    TimePoint time;
    TimePoint close_time;
    float open;
    float high;
    float low;
    float close;
    int tickvol;

    int label = -1;
};

class BarMarker : public App {
protected:

    std::vector<MinBar> m_allMinBars;
    std::unordered_map<String,float> m_pointValue;

    String m_symbol;
    BarMarker(const String& cfg);
public:
    virtual ~BarMarker() {;}

    static BarMarker& getInstance(const String& cfg) {
        static BarMarker _inst(cfg);
        return _inst;
    }

    void prepare();

    void execute();

    void finish();

    void parseBarFile(const String& barFile);

    void markMinBar();

    void dumpCSV();
};
#endif   /* ----- #ifndef _BAR_MARKER_H_  ----- */
