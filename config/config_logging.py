""" Configure Logging """
import os
import logging
import logging.config

import yaml


def logmaker(logname='run.log', path='./', mode='a', encoding=None):
    """ Set Default Path for File Loggers """
    log_path = os.path.join(path, logname)
    return logging.FileHandler(log_path, mode, encoding)


def setup_logging(default_path='./config/logging.yaml',
                  log_output_path='./',
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
        # change logging ouput paths
        for k, v in config['handlers'].items():
            if 'path' in v:
                v['path'] = log_output_path
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
