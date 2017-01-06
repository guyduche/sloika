SHELL=/bin/bash

pwd:=$(shell pwd)/
bin:=${pwd}bin/

#
# TODO(semen): currently setup-dev-env.sh creates a link but sloika is not picked up when running create_hdf5 below
#
.PHONY: train
train:
	${bin}create_hdf5.py --chunk 500 --kmer 5 --section template --strand_list train_strand_list.txt  fast5_directory dataset_train.hdf5

.PHONY: newEnv
newEnv: clean
	./setup-dev-env.sh

.PHONY: clean
clean:
	(source environment && rm -rf ${BUILD_DIR})
