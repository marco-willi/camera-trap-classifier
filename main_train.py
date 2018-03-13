""" Train a Model """
from config.config import Config
import logging
from logger import configure_logging

cfg = Config()
cfg.load_config()
configure_logging(logging, cfg.get_config())
cfg.get_config()


# 1 Import Configure
# 2 Set up logging
# Prepare data
# run model
# save model and print statistics
