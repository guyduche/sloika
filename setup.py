import re
from glob import glob
from setuptools import setup, find_packages

# Get the version number from __init__.py
package_name = 'sloika'
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
    'six',
    'tang', 
    'Theano'
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
    tests_require=requires,
    install_requires=requires,
    dependency_links=[],
    zip_safe=False,
    test_suite='discover_tests',
    scripts=[x for x in glob('bin/*.py') if x != 'bin/__init__.py'],
)
