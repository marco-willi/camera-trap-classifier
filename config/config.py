""" Parse config """
import os
import errno
import yaml
import logging
import sys
from datetime import datetime

##############################
# Config Class
##############################


class Config(object):
    def __init__(self, filename='config.yaml'):
        self.filename = filename
        self.ts = datetime.now().strftime('%Y%m%d%H%m')

    def load_config(self):
        self._load_from_disk()
        self._clean_paths()
        self._prepare_current_experiment()
        self._prepare_current_experiment()

    def _load_from_disk(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as fp:
                self.cfg = yaml.load(fp)
        else:
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT), self.filename)

    def _clean_paths(self):
        """ Clean paths and add separator """
        for v in self.cfg['paths'].values():
            os.path.normpath(v) + os.path.sep

    def _add_project_name_to_paths(config, project_name=''):
        """ Add project name to paths """
        for v in config['paths'].values():
            v = v + os.path.sep + project_name

    def _prepare_current_experiment(self):
        project = self.cfg['run']['project']
        exp = self.cfg['run']['experiment']
        exp_data = self.cfg['projects'][project]['experiments'][exp]
        project_data = self.cfg['projects'][project]
        self.current_project = project_data
        self.current_exp = exp_data
        for k, v in self.current_project.items():
            if not k == 'experiments':
                if k not in self.current_exp:
                    self.current_exp[k] = v



# Load Configuration
cfg = Config()
cfg.load_config()


##############################
# Logging
##############################

# logging handlers
handlers = list()

log_path = cfg.cfg['paths']['logging_output']

if cfg.cfg['general']['logging_to_disk']:
    # handlers to log stuff to (file and stdout)
    file_handler = logging.FileHandler(
        filename=log_path + 'run.log')
    handlers.append(file_handler)

stdout_handler = logging.StreamHandler(sys.stdout)
handlers.append(stdout_handler)

# logger configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(funcName)s - %(levelname)s:' +
                           '%(message)s',
                    handlers=handlers)

# log parameters / config
logging.info("Configuration: %s" % cfg.cfg)
