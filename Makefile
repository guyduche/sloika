SHELL=/bin/bash

pwd:=$(shell pwd)/
bin:=${pwd}bin/

sloikaVersion:=$(shell ./scripts/show-version.sh)
ifndef sloikaVersion
$(error $${sloikaVersion} is empty (not set))
endif
whlFile:=dist/sloika-${sloikaVersion}-cp27-cp27mu-linux_x86_64.whl

pipInstall:=pip install --trusted-host pypi.oxfordnanolabs.local --index-url https://pypi.oxfordnanolabs.local/simple/
inTmpEnv:=source environment && source $${TMP_VIRTUALENV_DIR}/bin/activate &&
inSloikaEnv:=source environment && source $${SLOIKA_VIRTUALENV_DIR}/bin/activate &&


.PHONY: test
test: unitTest acceptanceTest

#
# TODO(semen): not ideal that test requirements are installed into the same env where sloika is
#

unitTestCmd:=${pipInstall} -r test/unit/requirements.txt && cd test/unit && nose2
.PHONY: unitTest unitTestFromScratch
unitTest:
	${inSloikaEnv} ${unitTestCmd}
unitTestFromScratch: cleanTmpEnvWithSloika
	${inTmpEnv} ${unitTestCmd}

acceptanceTestCmd:=${pipInstall} -r test/acceptance/requirements.txt && cd test/acceptance && THEANO_FLAGS=$${THEANO_FLAGS_FOR_ACCTEST} nose2
.PHONY: acceptanceTest acceptanceTestFromScratch
acceptanceTest:
	${inSloikaEnv} ${acceptanceTestCmd}
acceptanceTestFromScratch: cleanTmpEnvWithSloika
	${inTmpEnv} ${acceptanceTestCmd}

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
	    libblas3 libblas-dev

.PHONY: checkout
checkout:
	git submodule init
	git submodule update


.PHONY: emptyTmpEnv
emptyTmpEnv:
	source environment && rm -rf $${TMP_VIRTUALENV_DIR} && virtualenv $${TMP_VIRTUALENV_DIR}
	${inTmpEnv} pip install pip --upgrade

.PHONY: wheel
wheel: emptyTmpEnv
	${inTmpEnv} ${pipInstall} -r scripts/requirements.txt && \
	            ${pipInstall} -r requirements.txt && \
	            python setup.py bdist_wheel

.PHONY: cleanTmpEnvWithSloika
cleanTmpEnvWithSloika: emptyTmpEnv
	${inTmpEnv} ${pipInstall} ${whlFile}






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
	(source environment && source $${SLOIKA_VIRTUALENV_DIR}/bin/activate && cd $${BUILD_DIR}/test && NOSE_PROCESSES=2 nosetests)
