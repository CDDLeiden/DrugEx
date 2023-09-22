"""
about

Created by: Martin Sicho
On: 24.06.22, 10:36
"""
import os

VERSION = "3.4.6"

if os.path.exists(os.path.join(os.path.dirname(__file__), '_version.py')):
    from ._version import version
    VERSION = version
