#!/usr/bin/env python

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup

setup(
    name="volta",
    version="0.0.1",
    author="Emanuele Bugliarello",
    description="",
    license="MIT",
    zip_safe=True,
    py_modules=['data', 'apex', 'volta', 'config', 'conversions', 'config_tasks', 'features_extraction']
)