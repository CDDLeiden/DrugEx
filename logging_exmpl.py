import os

import drugex.logs.utils
import utils
import logging_exmpl_mod
import logging
import argparse


def EnvironmentArgParser():
    """ 
        Define and read command line arguments
    """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-k', '--keep_runid', action='store_true', help="If included, continue from last run")
    parser.add_argument('-p', '--pick_runid', type=int, default=None, help="Used to specify a specific run id")
    parser.add_argument('-d', '--debug', action='store_true')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = EnvironmentArgParser()

    #Get run id
    runid = utils.get_runid(log_folder=os.path.join(args.base_dir,'logs'),
                            old=args.keep_runid,
                            id=args.pick_runid,
                            )

    #Configure logger
    utils.config_logger('%s/logs/%s/test.log' % (args.base_dir, runid), args.debug)

    #Get logger, include this in every module
    log = logging.getLogger(__name__)

    #Begin log file
    githash = drugex.logs.utils.commit_hash(os.path.dirname(os.path.realpath(__file__)))
    utils.init_logfile(log, runid, githash)
    
    log.debug('This message should go to the debug log file if debug argument is specified')
    log.debug('This message should not go to the debug log file ')
    log.info('This should be short format in only the log file')
    log.warning('This should be long format in log file and short in stream')
    log.error("Wow error")
    log.critical("Now it is serious")

    try:
        5/0
    except:
        log.exception("Oops dividing by zero is not possible")

    logging_exmpl_mod.test_func()