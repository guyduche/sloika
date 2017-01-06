SHELL=/bin/bash

pwd:=$(shell pwd)/
bin:=${pwd}bin/

#
# TODO(semen): currently setup-dev-env.sh creates a link but sloika is not picked up when running create_hdf5 below
#
.PHONY: train
train:
	${bin}create_hdf5.py --chunk 500 --kmer 5 --section template --strand_list train_strand_list.txt  fast5_directory dataset_train.hdf5

.PHONY: test
test:
	(source environment && source $${ACTIVATE} && python setup.py test)

#
# TODO(semen): fix parallel test runs
#              currently does not work because tests step on each other
#              in Theano intermediate directory
#
.PHONY: test_parallel
test_parallel:
	(source environment && source $${ACTIVATE} && NOSE_PROCESSES=2 nosetests .)

.PHONY: newEnv
newEnv: clean
	./setup-dev-env.sh

.PHONY: clean
clean:
	(source environment && rm -rf ${BUILD_DIR})
