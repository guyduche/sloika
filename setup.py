import re
from glob import glob
import numpy as np
import os
from setuptools import setup, find_packages
from Cython.Build import cythonize

# Get the version number from __init__.py
package_name = 'sloika'
package_dir = os.path.join(os.path.dirname(__file__), package_name)
verstrline = open('{}/__init__.py'.format(package_name), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(package_name))



# To ensure that we can build .debs with adequate dependencies,
#    .gitlab-ci.yml should contain package versions of these.
requires=[
    'numpy',
    'Theano',
    'untangled >= 0.3.1, < 0.4.0'
]

setup(
    name='sloika',
    version=verstr,
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
