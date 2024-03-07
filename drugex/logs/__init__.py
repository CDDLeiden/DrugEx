import logging
import sys

logger = None

if not logger:
    logger = logging.getLogger('drugex')
    logger.setLevel(logging.INFO)

def setLogger(log):
    setattr(sys.modules[__name__], 'drugex', log)