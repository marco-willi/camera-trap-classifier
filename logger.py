""" Set up logger """
import sys


def configure_logging(logging, config=None):
    """ Configure logging based on config """
    logging_handlers = list()

    # add file handler to log to disk
    if config is not None and config['general']['logging_to_disk']:
        file_handler = logging.FileHandler(
            filename=config['paths']['logging_output'] + 'run.log')
        logging_handlers.append(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    logging_handlers.append(stdout_handler)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(funcName)s - %(levelname)s:' +
                               '%(message)s',
                        handlers=logging_handlers)
