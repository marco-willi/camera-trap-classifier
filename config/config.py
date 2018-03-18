""" Parse config """
import os
import errno
import yaml
import logging
import sys

##############################
# Parse Config File
##############################

class Config():
    def __init__(self, filename='config.yaml'):
        self.filename = filename

    def load_config(self):
        self._load_from_disk()
        self._clean_paths()

    def get_config(self):
        return self.cfg

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



##############################
# Logging
##############################

# timestamp and logging file name
# ts = str(config['general']['ts'])
# if 'experiment_id' in cfg_model:
#     exp_id = cfg_model['experiment_id'] + '_'
# else:
#     exp_id = ''
#
# # logging handlers
handlers = list()
#
# if cfg_model['logging_to_disk'] == 1:
#     # handlers to log stuff to (file and stdout)
#     file_handler = logging.FileHandler(
#         filename=cfg_path['logs'] + exp_id + ts + '_run.log')
#     handlers.append(file_handler)

stdout_handler = logging.StreamHandler(sys.stdout)
handlers.append(stdout_handler)

# logger configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(funcName)s - %(levelname)s:' +
                           '%(message)s',
                    handlers=handlers)

# log parameters / config
#logging.info("Path Parameters: %s" % cfg_path)
#logging.info("Model Parameters: %s" % cfg_model)
