#!/usr/bin/env python

# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

from pathlib import Path
import re

from setuptools import setup, find_packages


parent_path = Path(__file__).parent

with (parent_path / 'pasha' / '__init__.py').open('r') as f:
    pattern = re.compile(r'^__version__ = \'(\d+\.\d+\.\d[a-z]*\d*)\'', re.M)

    for line in f:
        m = pattern.search(line)

        if m is not None:
            version = m.group(1)
            break
    else:
        raise RuntimeError('unable to find version string')


setup(
    name='pasha',
    version=version,
    description='Tools to parallelize operations on large data sets '
                'using shared memory with zero copies.',
    long_description=(parent_path / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    author='Philipp Schmidt',
    author_email='philipp.schmidt@xfel.eu',
    license='BSD-3-Clause',

    packages=find_packages(),

    python_requires='>=3.6',
    install_requires=['numpy'],

    classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Physics',
    ]
)
