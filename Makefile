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


.PHONY: autopep8
autopep8:
	${inSloikaEnv} autopep8 $f -i --max-line-length=120



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

n?=0
workDir?=run/$n/
fast5Dir?=/mnt/data/human/training/reads
strandTrain?=/mnt/data/human/training/na12878_train.txt
strandValidate?=/mnt/data/human/training/na12878_validation.txt
.PHONY: prepare
prepare:
	${inSloikaEnv} create_hdf5.py --chunk 500 --kmer 5 --section template --use_scaled \
	    --strand_list ${strandTrain} \
	    ${fast5Dir} ${workDir}dataset_train.hdf5
	${inSloikaEnv} create_hdf5.py --chunk 500 --kmer 5 --section template --use_scaled \
	    --strand_list ${strandValidate} \
	    ${fast5Dir} ${workDir}dataset_validate.hdf5

.PHONY: testPrepare
testPrepare:
	${inSloikaEnv} ${MAKE} prepare workDir:=$${BUILD_DIR}/prepare/ fast5Dir:=data/test_create_hdf5/reads/ \
	    strandTrain:=data/test_create_hdf5/na12878_train.txt \
	    strandValidate:=data/test_create_hdf5/na12878_train.txt

niteration?=50000
device?=gpu${gpu}
model?=models/baseline_gru.py
extraFlags?=
.PHONY: train
train:
	${inSloikaEnv} THEANO_FLAGS="${extraFlags}device=${device},$${COMMON_THEANO_FLAGS_FOR_TRAINING}" \
	    train_network.py --batch 100 --niteration ${niteration} --save_every 5000 --lrdecay 5000 --bad \
	    ${model} ${workDir}output ${workDir}dataset_train.hdf5

.PHONY: testTrain
testTrain:
	${inSloikaEnv} rm -rf $${BUILD_DIR}/prepare/output/
	${inSloikaEnv} ${MAKE} train workDir:=$${BUILD_DIR}/prepare/ \
	    niteration:=1 device:=cpu extraFlags:=profile=True, model:=models/tiny_gru.py

.PHONY: validate
validate:
	${inSloikaEnv} validate_network.py --bad --batch 200 ${model} ${workDir}dataset_validate.hdf5
