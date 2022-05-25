"""
__init__.py

Created by: Martin Sicho
On: 17.05.22, 9:53
"""
import logging

from rdkit import rdBase

rdBase.DisableLog('rdApp.info')
rdBase.DisableLog('rdApp.warning')

logger = logging.getLogger()