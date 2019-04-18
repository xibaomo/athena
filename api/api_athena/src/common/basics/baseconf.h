#ifndef BASECONF_H_INCLUDED
#define BASECONF_H_INCLUDED
#include "yaml-cpp/yaml.h"
#include "types.h"
#include "log.h"
#include "utils.h"
class YamlConfig {
protected:

    YAML::Node m_allConfig;
    String     m_cfgFile;

    YamlConfig(){;}
public:
    virtual ~YamlConfig() {;}

    void loadConfig(const String& cfgFile) {
        if(m_allConfig.IsNull()) {
            m_cfgFile = cfgFile;
            m_allConfig = YAML::LoadFile(cfgFile);
        } else {
            Log(LOG_FATAL) << "Yaml file already loaded. Overwrite not supported: " + m_cfgFile;
        }
    }

    template <typename T>
    T getKeyValue(const String& keys);

};

typedef YamlConfig BaseConfig;

template <typename T>
T
YamlConfig::getKeyValue(const String& keys)
{
    if (m_allConfig.IsNull()) {
        Log(LOG_FATAL) << "No yaml file loaded";
    }

    auto v = athena::splitString(keys,"/");
    const YAML::Node* p = &m_allConfig;
    for(size_t i = 0; i < v.size(); i++) {
        String& key = v[i];
        const auto& node = (*p)[key];
        if (!node) {
            Log(LOG_FATAL) << "Key not found: " + key;
        }
        p = &node;
        if (i == v.size()-1) {
            return p->as<T>();
        }
    }

    return T();
}

#endif // BASECONF_H_INCLUDED
