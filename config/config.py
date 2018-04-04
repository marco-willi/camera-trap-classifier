""" Parse config """
import os
import errno
import yaml
import logging
import sys
from datetime import datetime

from data_processing.utils import create_path, get_most_rescent_file_with_string


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
        self._prepare_current_paths()
        self._prepare_experiment(self.cfg['run']['location'],
                                 self.cfg['run']['experiment'])
        self._merge_default_model_settings()
        self._check_model_load_data()

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

        # best model save path
        best_model_path = model_path + 'model_best_' + self.run_id + \
                          id_postfix + '.hdf5'

        # check and create path if not exist
        for path in [run_dir, model_path]:
            create_path(path, create_path=True)

        # check path existence
        for path in [location_path, exp_data, exp_path, model_path, run_dir]:
            if not os.path.exists(path):
                raise FileNotFoundError("Path %s not found - create\
                                        prior to running code" % (path))
                                        
        paths = {'tfr_master': tfr_master_path,
                 'inventory': inventory_path,
                 'exp_data': exp_data,
                 'run_data': run_dir,
                 'model_saves': model_path,
                 'model_save_best': best_model_path,
                 'root': root}

        self.current_paths = paths

    def _merge_default_exp_settings(self):
        """ Merge default experiment settings """
        default_exp = self.cfg['locations']['default_config']['experiments']['default_config']
        for k, v in default_exp.items():
            if k not in self.current_exp:
                self.current_exp[k] = v

    def _merge_default_location_settings(self):
        """ Merge default experiment settings """
        default_loc = self.cfg['locations']['default_config']
        for k, v in default_loc.items():
            if k not in self.current_location and k is not 'experiments':
                self.current_location[k] = v

    def _merge_default_model_settings(self):
        """ Merge settings for models with experiment data """
        models = self.cfg['models']
        if self.current_exp['model'] not in models:
            raise IOError("Model %s not found in config file 'models'" %
                          self.current_exp['model'])

        model_settings = models[self.current_exp['model']]

        for setting, value in model_settings.items():
            if setting not in self.current_exp.keys():
                self.current_exp[setting] = value
            elif isinstance(value, dict):
                for setting_nested, value_nested in value.items():
                    if setting_nested not in self.current_exp[setting]:
                        self.current_exp[setting][setting_nested] = value_nested

    def _prepare_experiment(self, location, exp):
        """ Compile relevant information for current experiment """
        exp_data = self.cfg['locations'][location]['experiments'][exp]
        location_data = self.cfg['locations'][location]
        self.current_location = location_data
        self._merge_default_location_settings()
        self.current_exp = exp_data
        for k, v in self.current_location.items():
            if not k == 'experiments':
                if k not in self.current_exp:
                    self.current_exp[k] = v

        self._merge_default_exp_settings()

        self.current_model_loads = exp_data['load_model_from_disk']
        if self.current_model_loads['model_dir_to_load'] not in ('', None):

            root_model_path = self.current_paths['root'] + os.path.sep + \
                        self.current_model_loads['model_dir_to_load']

            if self.current_model_loads['model_file_to_load'] == 'latest':
                full_path = get_most_rescent_file_with_string(
                    dirpath=root_model_path,
                    in_str='.hdf5', excl_str='weights')
            else:
                full_path = root_model_path + \
                            os.path.sep + \
                            self.current_model_loads['model_file_to_load']
            self.current_model_loads['model_to_load'] = full_path
        else:
            self.current_model_loads['model_to_load'] = ''

    def _check_model_load_data(self):
        if all([self.current_model_loads['continue_training'],
                self.current_model_loads['transfer_learning']]):
            ImportError("load_model_from_disk configuration: only one of \
                        transfer_learning and continue training can be true")

        if any([self.current_model_loads['continue_training'],
                self.current_model_loads['transfer_learning']]):
            if not os.path.isfile(self.current_model_loads['model_to_load']):
                FileNotFoundError("Model file to load %s was not found" % \
                                   self.current_model_loads['model_to_load'])


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
