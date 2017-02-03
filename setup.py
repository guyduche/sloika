import re
from glob import glob
import numpy as np
import os
import subprocess
from setuptools import setup, find_packages
from Cython.Build import cythonize

package_name = 'sloika'
package_dir = os.path.join(os.path.dirname(__file__), package_name)

cmd = './scripts/show-version.sh'
version, err = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE).communicate()
open('sloika/sloika_version.py','w').write("__version__ = '%s'\n"%version)

install_requires = [
'h5py==2.6.0',
'numpy>=1.7.1',
'Theano==0.8.2',
'untangled>=0.5.0',
]

setup(
    name='sloika',
    version=version,
    description='Theano RNN library',
    maintainer='Tim Massingham',
    maintainer_email='tim.massingham@nanoporetech.com',
    url='http://www.nanoporetech.com',
    long_description="""Something to do with sheep""",
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test", "bin"]),
    package_data={'configs':'data/configs/*'},
    exclude_package_data={'': ['*.hdf', '*.c', '*.h']},
    ext_modules = cythonize(os.path.join(package_dir, "viterbi_helpers.pyx")),
    include_dirs=[np.get_include()],
    tests_require=[],
    install_requires=install_requires,
    dependency_links=[],
    zip_safe=False,
    test_suite='discover_tests',
    scripts=[x for x in glob('bin/*.py') if x != 'bin/__init__.py'],
)
