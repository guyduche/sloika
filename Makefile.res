n?=0
workDir?=run/$n
fast5Dir?=/mnt/data/human/training/reads
strandTrain?=/mnt/data/human/training/na12878_train.txt
strandValidate?=/mnt/data/human/training/na12878_validation.txt

.PHONY: prepare
prepare:
	${inDevEnv} $${BIN_DIR}/chunkify.py identity --chunk_len 500 --kmer_len 5 --section template --jobs 1 --overwrite \
	    --input_strand_list ${strandTrain} \
	    ${fast5Dir} ${workDir}/dataset_train.hdf5
	${inDevEnv} $${BIN_DIR}/chunkify.py identity --chunk_len 500 --kmer_len 5 --section template --jobs 1 --overwrite \
	    --input_strand_list ${strandValidate} \
	    ${fast5Dir} ${workDir}/dataset_validate.hdf5

.PHONY: testPrepare
testPrepare:
	${inDevEnv} ${MAKE} prepare workDir:=$${BUILD_DIR}/prepare/ fast5Dir:=data/test_chunkify/identity/reads/ \
	    strandTrain:=data/test_chunkify/identity/na12878_train.txt \
	    strandValidate:=data/test_chunkify/identity/na12878_train.txt

.PHONY: remap
remap:
	${inDevEnv} $${BIN_DIR}/chunkify.py remap --chunk_len 500 --kmer_len 5 --section template --jobs 1 --overwrite \
	    --input_strand_list ${strandTrain} \
	    ${fast5Dir} ${workDir}/dataset_train.hdf5 data/test_chunkify/remap/model.pkl data/test_chunkify/remap/reference.fa

.PHONY: testRemap
testRemap:
	${inDevEnv} ${MAKE} remap workDir:=$${BUILD_DIR}/prepare/ fast5Dir:=data/test_chunkify/identity/reads/ \
	    strandTrain:=data/test_chunkify/identity/na12878_train.txt \
	    strandValidate:=data/test_chunkify/identity/na12878_train.txt

d:=/media/scratch/dnewman/results/rna_training/rnn_new_ev/
.PHONY: remap2
remap2:
	${inDevEnv} THEANO_FLAGS=$${THEANO_FLAGS_FOR_ACCTEST} chunkify.py remap \
	    data/test_chunkify/remap2/reads \
	    remap_reads.hdf5 \
	    ${d}sloika_train_v2/models/model_final.pkl \
	    ${d}remap_reads/refseqs.fasta \
	    --input_strand_list data/test_chunkify/remap2/strand_input_list.txt --overwrite

niteration?=50000
device?=gpu${gpu}
model?=models/baseline_gru.py
extraFlags?=
.PHONY: train
train:
	${inDevEnv} THEANO_FLAGS="${extraFlags}device=${device},$${COMMON_THEANO_FLAGS_FOR_TRAINING}" \
	    train_network.py --batch 100 --niteration ${niteration} --save_every 5000 --lrdecay 5000 --bad \
	    ${model} ${workDir}/output ${workDir}/dataset_train.hdf5

.PHONY: testTrain
testTrain:
	${inDevEnv} rm -rf $${BUILD_DIR}/prepare/output/
	${inDevEnv} ${MAKE} train workDir:=$${BUILD_DIR}/prepare/ \
	    niteration:=1 device:=cpu extraFlags:=profile=True, model:=models/tiny_gru.py

.PHONY: validate
validate:
	${inDevEnv} validate_network.py --bad --batch 200 ${model} ${workDir}/dataset_validate.hdf5
