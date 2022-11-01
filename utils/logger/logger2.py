import os, yaml
import logging.config
import params


# Wrapper for the default logger
# Allow to set the logging path and the logging mode from a config file

C_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_CONF = None
try:
    LOG_CONF = params.LOG_CON
except:
    pass
if LOG_CONF == None:
    params.LOG_CONF = "%s/logger.yaml" % C_DIR


class MyLogger():
    def __init__(self, logPath=None):
        with open(params.LOG_CONF) as f:
            D = yaml.load(f, Loader=yaml.FullLoader)
            # print D
            D.setdefault('version', 1)
        if logPath != None:
            D['handlers']['file']['filename'] = logPath

        logging.config.dictConfig(D)

        # create logger
        self.allLogger = logging.getLogger('allLogger')
        self.fileLogger = logging.getLogger('fileLogger')
        self.consoleLogger = logging.getLogger('consoleLogger')

    def infoAll(self, msg):
        self.allLogger.info(msg)

    def infoFile(self, msg):
        self.fileLogger.info(msg)
