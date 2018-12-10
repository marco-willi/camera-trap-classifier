""" Parse config """
import os
import errno
import yaml


##############################
# Config Class
##############################

class ConfigLoader(object):
    def __init__(self, filename='config.yaml'):
        self.filename = filename
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as fp:
                self.cfg = yaml.load(fp)
        else:
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT), self.filename)
