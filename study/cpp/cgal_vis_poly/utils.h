/*
 * =====================================================================================
 *
 *       Filename:  utils.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/19/2019 11:24:19 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _COMMON_UTILS_H_
#define  _COMMON_UTILS_H_
#include <ctime>
#include <chrono>

class MsTimer {
private:
    std::chrono::high_resolution_clock::time_point m_start;
public:
    MsTimer() { restart(); }
    void restart() {
        m_start = std::chrono::high_resolution_clock::now();
    }

    double elapsed() {
        std::chrono::duration<double, std::milli> dt = std::chrono::high_resolution_clock::now() - m_start;
        return dt.count();
    }
};
#endif   /* ----- #ifndef _COMMON_UTILS_H_  ----- */
