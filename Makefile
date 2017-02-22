SHELL=/bin/bash

pwd:=$(shell pwd)/
bin:=${pwd}bin/
nproc:=$(shell nproc)

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
# TODO(semen): do not install unit and acctest deps into the same env where sloika is
#

unitset?=
unitTestCmd:=${pipInstall} -r test/unit/requirements.txt && cd test/unit && py.test -n auto ${unitset}
.PHONY: unitTest unitTestFromScratch
unitTest:
	${inSloikaEnv} ${unitTestCmd}
unitTestFromScratch: cleanTmpEnvWithSloika
	${inTmpEnv} ${unitTestCmd}

accset?=
acceptanceTestCmd:=${pipInstall} -r test/acceptance/requirements.txt && cd test/acceptance && py.test -n auto ${accset}
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
	    libblas3 libblas-dev python-dev python3-dev

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

pyDirs:=sloika test bin models misc
pyFiles:=$(shell find *.py ${pyDirs} -type f -regextype sed -regex ".*\.py")
.PHONY: autopep8
autopep8:
	${inSloikaEnv} autopep8 --ignore E203 -i --max-line-length=120 ${pyFiles}

.PHONY: pep8
pep8:
	${inSloikaEnv} pep8 --ignore E203,E402 --max-line-length=120 ${pyFiles}

cmd?=echo "Set 'cmd' to command to run in Sloika env"
.PHONY: runInEnv
runInEnv:
	@${inSloikaEnv} ${cmd}

.PHONY: pp pullpush
pp: pullpush
pullpush:
	${MAKE} pep8
	cd data && git pull --rebase && git push
	git pull --rebase && git push

include Makefile.res
