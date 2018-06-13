""" Configure Logging """
import os
import logging
import logging.config

import yaml


def logmaker(logname='run.log', path='./', mode='a', encoding=None):
    """ Set Default Path for File Loggers """
    return logging.FileHandler(path + logname, mode, encoding)


def setup_logging(default_path='./config/logging.yaml',
                  default_level=logging.DEBUG,
                  env_key='LOG_CFG'):
    """ Setup logging configuration """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
