# Copyright 2024 The SeqIO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install SeqIO."""

import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'seqio')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setuptools.setup(
    name='seqio-nightly',
    version=__version__,
    description=(
        'SeqIO: Task-based datasets, preprocessing, and evaluation for sequence'
        ' models.'
    ),
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='https://github.com/google/seqio/tree/nightly',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    include_package_data=True,
    scripts=[],
    install_requires=[
        'absl-py',
        'clu',
        'editdistance',
        'jax',
        'jaxlib',
        'numpy',
        'packaging',
        'pyglove',
        'sentencepiece >= 0.2',
        'tensorflow-text',
        # Ping to a specific version to avoid endless backtracking during
        # pip dependency resolution.
        'tfds-nightly==4.9.2.dev202308090034',
    ],
    requires_python='>=3.9',
    extras_require={
        'gcp': [
            'gevent',
            'google-api-python-client',
            'google-compute-engine',
            'google-cloud-storage',
            'oauth2client',
        ],
        'cache-tasks': ['apache-beam'],
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'seqio_cache_tasks ='
            ' seqio.scripts.cache_tasks_main:console_entry_point'
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='sequence preprocessing nlp machinelearning',
)
