#!/usr/bin/env python
'''
Created on Sep 3, 2018

@author: fxua
'''


import sys
import os 
from modules.basics.common.logger import *
from modules.basics.conf.masterconf import gMasterConfig
from modules.basics.conf.generalconf import gGeneralConfig
from apps.appcreator import createApp
from modules.cmdlineparser.cmdlineparser import CmdLineParser

if __name__ == "__main__":
    if len(sys.argv) < 2:
        Log(LOG_FATAL) << "No Yaml file is provided"
        sys.exit()
    
    gMasterConfig.loadLocalConfig(sys.argv[1])
    gMasterConfig.prepare()
    Log.setlogLevel(gGeneralConfig.getLogLevel())
    cmdParser = CmdLineParser()
    gGeneralConfig.loadCmdLineOpts(cmdParser.getParsedArgs())
    Log(LOG_INFO) << "Program starts ..."
    
    app = createApp(gGeneralConfig.getAppType())
    
    app.prepare()
    
    app.execute()
    
    app.finish()
    
    Log(LOG_INFO) << "Program exits normally."