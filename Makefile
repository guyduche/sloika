SHELL=/bin/bash

pwd:=$(shell pwd)/
bin:=${pwd}bin/

# TODO(semen): sort out versioning
whlFile:=dist/sloika-1.1.dev0-cp27-cp27mu-linux_x86_64.whl

# this target can only be run in serial
.PHONY: testFromScratch
testFromScratch: cleanVirtualenv install test

#
# TODO: can't run tests reliably from the tree where source directory is named sloika
#
.PHONY: test
test:
	(source environment && rm -rf $${BUILD_DIR}/test)
	(source environment && cp -r sloika/test $${BUILD_DIR})
	(source environment && source $${ACTIVATE} && cd $${BUILD_DIR}/test && nose2)

#
# TODO(semen): fix parallel test runs
#              currently does not work because tests step on each other
#              in Theano intermediate directory
#
# TODO(semen): upgrade to nose2
#
.PHONY: test_parallel
test_parallel:
	(source environment && source $${ACTIVATE} && NOSE_PROCESSES=2 nosetests .)

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

.PHONY: deps
deps:
	apt-get update
	apt-get install -y \
	    python-virtualenv python-pip python-setuptools ont-ca-certs git

.PHONY: wheel
wheel: ${whlFile}
${whlFile}: setup.py Makefile
	(source environment && source $${ACTIVATE} && python setup.py bdist_wheel)

.PHONY: install
install: ${whlFile}
	(source environment && source $${ACTIVATE} && pip install ${whlFile})
