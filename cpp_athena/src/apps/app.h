/*
 * =====================================================================================
 *
 *       Filename:  app.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/02/2018 02:51:42 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _APPS_APP_H_
#define  _APPS_APP_H_
#include "basics/utils.h"
enum class AppType {
    BAR_MARKER = 0,
};

class App {
protected:
    String m_yamlParser;
    String m_configFile;
public:
    App(const String& config) : m_configFile(config) { m_yamlParser = String(getenv("ATHENA_INSTALL")) + "/api/release/scripts/yaml_parser.py ";}
    virtual ~App() {;}

    virtual void prepare() = 0;

    virtual void execute() = 0;

    virtual void finish()  = 0;

    String getYamlValue(const String& key) {
        String cmd = m_yamlParser + key + " " + m_configFile;
        String val = execSysCall_block(cmd);
        return val;
    }
};

App* createApp(AppType atp, const String& cfg);
#endif   /* ----- #ifndef _APPS_APP_H_  ----- */
