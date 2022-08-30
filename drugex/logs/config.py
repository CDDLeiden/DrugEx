import os
from datetime import datetime
import logging
from logging import config
from bisect import bisect
from typing import Dict

class LevelFilter(logging.Filter):
    """
        LoggingFilter used to filter one or more specific log levels messages
    """
    def __init__(self, level):
        self.__level = level

    def filter(self, record):
        return record.levelno in self.__level 

#Adapted from https://stackoverflow.com/a/68154386
class LevelFormatter(logging.Formatter):
    """
        LoggingFormatter used to specifiy the formatting per level
    """
    def __init__(self, formats: Dict[int, str], **kwargs):
        super().__init__()

        if 'fmt' in kwargs:
            raise ValueError(
                'Format string must be passed to level-surrogate formatters, '
                'not this one'
            )

        self.formats = sorted(
            (level, logging.Formatter(fmt, **kwargs)) for level, fmt in formats.items()
        )

    def format(self, record: logging.LogRecord) -> str:
        idx = bisect(self.formats, (record.levelno,), hi=len(self.formats)-1)
        level, formatter = self.formats[idx]
        return formatter.format(record)

def config_logger(log_file_path, debug=None, disable_existing_loggers=True):
    """
        Function to configure the logging. 
        All info is saved in a simple format on the log file path.
        Debug entries are saved to a separate file if debug is True
        Debug and warning and above are save in a verbose format.
        Warning and above are also printed to std.out
        ...

        Arguments
        ----------
        log_file_path (str): Folder where all logs for this run are saved
        debug (bool): if true, debug messages are saved
        no_exist_log (bool): if true, existing loggers are disabled
    """
    debug_path = os.path.join(os.path.dirname(log_file_path), 'debug.log')
    simple_format = '%(message)s'
    verbose_format = '[%(asctime)s] %(levelname)s [%(filename)s %(name)s %(funcName)s (%(lineno)d)]: %(message)s'

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': disable_existing_loggers,
        'formatters': {
            'simple_formatter': {
                'format': simple_format
            },
            'verbose_formatter': {
                'format': verbose_format
            },
            'bylevel_formatter':{
                '()':  LevelFormatter,
                'formats': {
                                logging.DEBUG: verbose_format,
                                logging.INFO: simple_format,
                                logging.WARNING: verbose_format
                            }
            }
        },
        'filters': {
            'only_debug': {
                '()': LevelFilter,
                'level': [logging.DEBUG]
            }
        },
        'handlers': {
            'stream_handler': {
                'class': 'logging.StreamHandler',
                'formatter': 'simple_formatter',
                'level': 'WARNING'
            },
            'file_handler': {
                'class': 'logging.FileHandler',
                'formatter': 'bylevel_formatter',
                'filename': log_file_path,
                'level': 'INFO'
            },
            'file_handler_debug': {
                'class': 'logging.FileHandler',
                'formatter': 'bylevel_formatter',
                'filename': debug_path,
                'mode': 'w',
                'delay': True,
                'filters': ['only_debug']
            },
        },
        'loggers': {
            None: {
                'handlers': ['stream_handler', 'file_handler', 'file_handler_debug'] \
                            if debug else ['stream_handler', 'file_handler'],
                'level': 'DEBUG'
            }
        }
    }
    
    config.dictConfig(LOGGING_CONFIG)

def init_logfile(log, githash=None, args=None):
    """
        Put some intial information in the logfile
        ...

        Arguments
        ----------
        log : Logging instance
        runid (str): the current runid
        githash (str): githash
    """
    if os.path.getsize(log.root.handlers[1].baseFilename) == 0:
        logging.info('Creation date: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    else: 
        logging.info('\nContinued at: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logging.info('git hash: %s \n' % githash)
    logging.info('Run settings:')
    logging.info(args)
    logging.info('')

def get_runid(log_folder='logs', old=True, id=None):
    """
        Fetch runid that is used in all logfiles to identifiy a specific run
        ...

        Arguments
        ----------
        log_folder (str): Folder where all logs are saved
        old (bool): if true, fetches the last used runid
        id (int): If included, returns this runid number
    """

    fname = os.path.join(log_folder, 'runid.txt')

    # Determine the runid to fetch
    if not id is None:
        runid = id
    else:
        runid = 1
        if os.path.isfile(fname):
            with open(fname, 'r') as f:
                prev_runid = int(f.readlines()[-1][-4:])
                if old:
                    runid = prev_runid
                else:
                    runid = prev_runid + 1

    #Maximum runid is 9999
    if runid >= 10000 or runid <= 0:
        raise ValueError('Run id larger than 9999 or smaller than 1')

    #Create log directory if necessary
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)

    #Write runid to runid file
    with open(fname, 'a') as f:
        f.write('\n%s: %04d' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), runid))

    #Create runid log directory if necessary
    if not os.path.isdir(os.path.join(log_folder, '%04d' % runid)):
        os.mkdir(os.path.join(log_folder, '%04d' % runid))

    return '%04d' % runid

class LogFileConfig:

    def __init__(self, path, logger, debug):
        self.path = path
        self.log = logger
        self.debug = debug