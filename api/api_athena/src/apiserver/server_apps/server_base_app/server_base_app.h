/*
 * =====================================================================================
 *
 *       Filename:  server_base_app.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/14/2018 01:36:19 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef _API_SERVER_BASE_APP_H_
#define _API_SERVER_BASE_APP_H_

#include "app_base/app_base.h"
#include "basics/types.h"
#include "messenger/messenger.h"
#include "basics/utils.h"
enum class EngineType {
    CLASSIFIER = 0,
    REGRESSOR
};

enum class EngineCoreType {
    SVM = 0,
    DCT,
    RANDOMFOREST
};
class ServerBaseApp : public App {
protected:
    String m_configFile;
    String m_yamlParser;
    ServerBaseApp(const String& configFile) : m_configFile(configFile) {
        m_yamlParser = String(getenv("ATHENA_INSTALL")) + "/api/release/scripts/yaml_parser.py ";
    }

public:
    virtual ~ServerBaseApp() {;}

    virtual void prepare() {BASE_METHOD_WARN;}

    void execute();

    virtual void finish() {;}

    virtual Message processMsg(Message& msg) { BASE_METHOD_WARN; return Message();}

    String getYamlValue(const String& key) {
        String cmd = m_yamlParser + key + " " + m_configFile;
        String val = athena::execSysCall_block(cmd);
        return val;
    }

    template <typename Fn>
    Message procMsg_noreply(Message& msg, const Fn& fn) {
        fn(msg);
        Message out;
        return out;
    }

};

#endif
