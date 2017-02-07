n?=0
workDir?=run/$n/
fast5Dir?=/mnt/data/human/training/reads
strandTrain?=/mnt/data/human/training/na12878_train.txt
strandValidate?=/mnt/data/human/training/na12878_validation.txt

.PHONY: prepare
prepare:
	${inSloikaEnv} $${BIN_DIR}/chunkify.py identity --chunk_len 500 --kmer_len 5 --section template --use_scaled --threads 1 \
	    --strand_list ${strandTrain} \
	    ${fast5Dir} ${workDir}dataset_train.hdf5
	${inSloikaEnv} $${BIN_DIR}/chunkify.py identity --chunk_len 500 --kmer_len 5 --section template --use_scaled --threads 1 \
	    --strand_list ${strandValidate} \
	    ${fast5Dir} ${workDir}dataset_validate.hdf5

.PHONY: testPrepare
testPrepare:
	${inSloikaEnv} ${MAKE} prepare workDir:=$${BUILD_DIR}/prepare/ fast5Dir:=data/test_chunkify/identity/reads/ \
	    strandTrain:=data/test_chunkify/identity/na12878_train.txt \
	    strandValidate:=data/test_chunkify/identity/na12878_train.txt

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
