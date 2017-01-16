SHELL=/bin/bash

pwd:=$(shell pwd)/
bin:=${pwd}bin/

# TODO(semen): sort out versioning
version:=$(shell python scripts/version.py)
whlFile:=dist/sloika-${version}-cp27-cp27mu-linux_x86_64.whl

# these targets can only be run in serial
.PHONY: testFromScratch unitTestFromScratch acceptanceTestFromScratch unitTestFromScratchInParallel
testFromScratch: cleanVirtualenv install unitTest acceptanceTest
unitTestFromScratch: cleanVirtualenv install unitTest
acceptanceTestFromScratch: cleanVirtualenv install acceptanceTest
unitTestFromScratchInParallel: cleanVirtualenv install testInParallel

.PHONY: test
test: unitTest acceptanceTest

.PHONY: acceptanceTest acctest
acceptanceTest: acctest
acctest:
	(source environment && source $${ACTIVATE} && cd test/acceptance && THEANO_FLAGS=$${ACCTEST_TF} nose2)

.PHONY: unitTest unit
unit: unitTest
unitTest:
	(source environment && source $${ACTIVATE} && cd test/unit && nose2)

#
# TODO(semen): fix parallel test runs
#              currently does not work because tests step on each other
#              in Theano intermediate directory
#
# TODO(semen): upgrade to nose2
#
.PHONY: testInParallel
testInParallel:
	(source environment && rm -rf $${BUILD_DIR}/test)
	(source environment && cp -r sloika/test $${BUILD_DIR})
	(source environment && source $${ACTIVATE} && cd $${BUILD_DIR}/test && NOSE_PROCESSES=2 nosetests)

.PHONY: cleanCiEnv
cleanCiEnv: cleanVirtualenv ${whlFile}
	(source environment && source $${ACTIVATE} && pip install ${whlFile})

.PHONY: cleanDevEnv
cleanDevEnv: cleanVirtualenv
	./setup-dev-env.sh

.PHONY: cleanVirtualenv
cleanVirtualenv: clean
	./setup-virtualenv.sh

.PHONY: clean
clean:
	(source environment && rm -rf $${BUILD_DIR})
	rm -rf dist

.PHONY: deps
deps:
	apt-get update
	apt-get install -y \
	    python-virtualenv python-pip python-setuptools ont-ca-certs git \
	    python-yaml libblas3 libblas-dev

.PHONY: checkout
checkout:
	git submodule init
	git submodule update

.PHONY: wheel
wheel: ${whlFile}
${whlFile}: setup.py Makefile
	(source environment && source $${ACTIVATE} && python setup.py bdist_wheel)

.PHONY: install
install: ${whlFile}
	(source environment && source $${ACTIVATE} && pip install ${whlFile})
