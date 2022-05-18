"""
utils

Created by: Martin Sicho
On: 17.05.22, 9:55
"""
import json
import logging
import os

import git

from drugex.logs import config
from drugex.logs.config import LogFileConfig


def commit_hash(GIT_PATH):
    repo = git.Repo.init(GIT_PATH)
    return '#' + repo.head.object.hexsha[:8]

def enable_file_logger(log_folder, filename, keep_old_runid=False, picked_runid=None, debug=False, log_name=None, git_hash=None, init_data=None):

    # Get run id
    runid = config.get_runid(log_folder=log_folder,
                            old=keep_old_runid,
                            id=picked_runid)
    path = os.path.join(log_folder, f'{runid}/{filename}')
    config.config_logger(path, debug)

    # get logger and init configuration
    log = logging.getLogger(filename) if not log_name else logging.getLogger(log_name)
    settings = LogFileConfig(path, log, debug, runid)

    # Begin log file
    config.init_logfile(log, runid, git_hash, json.dumps(init_data, sort_keys=False, indent=2))

    return settings