""" Parse config """
import os
import errno
import yaml
import logging
import sys
from datetime import datetime

from data_processing.utils import create_path

##############################
# Config Class
##############################

class Config(object):
    def __init__(self, filename='config.yaml'):
        self.filename = filename
        self.ts = datetime.now().strftime('%Y%m%d%H%m')
        self.run_id= 'run_' + self.ts

    def load_config(self):
        self._load_from_disk()
        self._clean_paths()
        self._prepare_current_experiment()
        self._prepare_current_paths()

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

    def _prepare_current_paths(self):
        """ Create Paths and Directories """
        # general paths
        root = self.cfg['paths']['root']
        exp = root + self.cfg['paths']['experiments']
        models = root + self.cfg['paths']['models']

        # project specific paths
        project = self.cfg['run']['project']
        project_path = root + project + os.path.sep
        tfr_master_file = self.cfg['projects'][project]['paths']['master_tfr']
        tfr_master_path = project_path + 'data' + os.path.sep + tfr_master_file
        inventory_file = self.cfg['projects'][project]['paths']['inventory']
        inventory_path = project_path + 'data' + os.path.sep + inventory_file

        # experiment specific paths
        exp_path = exp + self.cfg['run']['experiment'] + os.path.sep
        model_path = models + self.cfg['run']['experiment'] + os.path.sep

        exp_data = exp_path + 'data' + os.path.sep
        run_dir = exp_path + self.run_id + os.path.sep

        # # check and create path if not exist
        # for path in [exp_data, run_dir]:
        #     create_path(path, create_path=True)
        paths = {'tfr_master': tfr_master_path,
                 'inventory': inventory_path,
                 'exp_data': exp_data,
                 'run_data': run_dir,
                 'model_saves': model_path}

        self.current_paths = paths

    def _prepare_current_experiment(self):
        """ Compile relevant information for current experiment """
        project = self.cfg['run']['project']
        exp = self.cfg['run']['experiment']
        model = self.cfg['run']
        exp_data = self.cfg['projects'][project]['experiments'][exp]
        model = exp_data['model']
        model_cfg = self.cfg['models'][model]
        project_data = self.cfg['projects'][project]
        self.current_project = project_data
        self.current_exp = exp_data
        self.current_model = model_cfg
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

#log_path = cfg.cfg['paths']['logging_output']
log_path = ''

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
