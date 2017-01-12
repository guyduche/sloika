import re
from glob import glob
import numpy as np
import os
import yaml
import subprocess
from setuptools import setup, find_packages
from Cython.Build import cythonize

package_name = 'sloika'
package_dir = os.path.join(os.path.dirname(__file__), package_name)

cmd = 'python scripts/version.py'
out, err = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE).communicate()
version = out.strip()
open('sloika/sloika_version.py','w').write("__version__ = '%s'\n"%version)

requires=[]

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
    tests_require=requires,
    install_requires=requires,
    dependency_links=[],
    zip_safe=False,
    test_suite='discover_tests',
    scripts=[x for x in glob('bin/*.py') if x != 'bin/__init__.py'],
)
