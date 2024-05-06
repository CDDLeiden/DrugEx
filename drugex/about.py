import os

VERSION = "3.4.7"

if os.path.exists(os.path.join(os.path.dirname(__file__), '_version.py')):
    from ._version import version
    VERSION = version
