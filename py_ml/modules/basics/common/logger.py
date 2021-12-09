'''
Created on Sep 3, 2018

@author: xfb
'''
import logging
import inspect
import sys
LOG_FATAL = 0
LOG_ERROR = 1
LOG_INFO = 2
LOG_DEBUG = 3
LogSwitcher = {
        0: logging.fatal,
        1: logging.error,
        2: logging.info,
        3: logging.debug
}
class Logger(object):
    def __init__(self):
        self.logLevel = LOG_DEBUG
        return
    
    def __call__(self, loglevel):
        if loglevel > self.logLevel:
            self.isActive = False
            return self

        self.isActive = True
        self.requestLevel = loglevel
        self.func = LogSwitcher.get(loglevel, "no such a log level")
        return self

    def __lshift__(self, msg):
        if not self.isActive:
            return
        file_lineno=""
        if self.requestLevel == LOG_FATAL or self.requestLevel == LOG_ERROR:
            caller = inspect.stack()[1]
            frame = caller[0]
            info = inspect.getframeinfo(frame)
            file_lineno= "["+info.filename+":"+ str(info.lineno)+"]"
        
        self.func(file_lineno + msg)
        if self.requestLevel == LOG_FATAL:
            sys.exit()
        return

    def setlogLevel(self, loglevel):
        self.logLevel = loglevel
        logging.StreamHandler(sys.stdout)
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s %(message)s',datefmt='%m-%d-%Y:%H:%M:%S',level=loglevel)
        return
    
global Log
Log = Logger()
