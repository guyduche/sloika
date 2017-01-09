SHELL=/bin/bash

pwd:=$(shell pwd)/
bin:=${pwd}bin/

.PHONY: testFromScratch
testFromScratch: cleanVirtualenv
	(source environment && source $${ACTIVATE} && nose2)

.PHONY: test
test:
	(source environment && source $${ACTIVATE} && nose2)

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
	    python-virtualenv python-pip python-setuptools ont-ca-certs


# unused / experimental targets follow
#
.PHONY: wheel
wheel:
	pip wheel .

.PHONY: wheeldeps
wheeldeps:
	pip wheel -r requirements.txt --trusted-host pypi.oxfordnanolabs.local \
	   -i https://pypi.oxfordnanolabs.local/simple/
