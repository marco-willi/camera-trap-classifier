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

    def _add_location_name_to_paths(config, location_name=''):
        """ Add location name to paths """
        for v in config['paths'].values():
            v = v + os.path.sep + location_name

    def _prepare_current_paths(self):
        """ Create Paths and Directories """
        # general paths
        if self.cfg['general']['debug']:
            paths = 'paths_debug'
        else:
            paths = 'paths'

        root = self.cfg[paths]['root']
        exp = self.cfg[paths]['experiments']
        models = self.cfg[paths]['models']

        # location specific paths
        location = self.cfg['run']['location']
        location_path = root + location + os.path.sep
        tfr_master_file = self.cfg['locations'][location]['paths']['master_tfr']
        tfr_master_path = location_path + 'data' + os.path.sep + tfr_master_file
        inventory_file = self.cfg['locations'][location]['paths']['inventory']
        inventory_path = location_path + 'data' + os.path.sep + inventory_file

        # experiment specific paths
        exp_path = location_path + exp + self.cfg['run']['experiment'] + os.path.sep
        model_path = location_path + models + self.cfg['run']['experiment'] + os.path.sep

        exp_data = exp_path + 'data' + os.path.sep

        id_postfix = self.cfg['run']['identifier_postfix']
        run_dir = exp_path + self.run_id + id_postfix + os.path.sep

        # check and create path if not exist
        for path in [run_dir]:
            create_path(path, create_path=True)
        paths = {'tfr_master': tfr_master_path,
                 'inventory': inventory_path,
                 'exp_data': exp_data,
                 'run_data': run_dir,
                 'model_saves': model_path,
                 'root': root}

        self.current_paths = paths

    def _prepare_current_experiment(self):
        """ Compile relevant information for current experiment """
        location = self.cfg['run']['location']
        exp = self.cfg['run']['experiment']
        exp_data = self.cfg['locations'][location]['experiments'][exp]
        location_data = self.cfg['locations'][location]
        self.current_location = location_data
        self.current_exp = exp_data
        for k, v in self.current_location.items():
            if not k == 'experiments':
                if k not in self.current_exp:
                    self.current_exp[k] = v

##############################
# Logging
##############################


def configure_logging(cfg):
    # logging handlers
    handlers = list()

    log_path = cfg.current_paths['run_data']

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

    return logging
