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

#include "ma_hunter.h"

using namespace std;
using namespace athena;

MAHunter::MAHunter(const String& cfg) : m_config(nullptr), ServerBaseApp(cfg)
{
    m_config = &MahuntConfig::getInstance();
    m_config->loadConfig(cfg);
}
