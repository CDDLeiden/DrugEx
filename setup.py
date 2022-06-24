"""
setup.py

Created by: Martin Sicho
On: 24.06.22, 10:33
"""

from setuptools import setup
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('drugex/about.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(version=main_ns['VERSION'])