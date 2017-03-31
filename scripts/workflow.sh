#! /bin/bash -eu

apt-get install bwa

# move to sloika top-level dir
SLOIKA_ROOT=$(git rev-parse --show-toplevel)

(cd $SLOIKA_ROOT && make checkout && make cleanDevEnv)
source sloika/build/env/bin/activate
pip install pysam matplotlib

# Create working directory
$WORK_DIR=$SLOIKA_ROOT/build/workflow
mkdir -p $WORK_DIR && cd $WORK_DIR

READ_DIR=$SLOIKA_ROOT/data/test_chunkify/identity/reads
REFERENCE=$SLOIKA_ROOT/data/test_chunkify/identity/reference.fa
MODEL=$SLOIKA_ROOT/data/test_basecall_network/raw_model_1pt2_cpu.pkl


# 1. Basecall with existing model
export OMP_NUM_THREADS=1
export THEANO_FLAGS=device=cpu,floatX=float32
$SLOIKA_ROOT/bin/basecall_network.py raw $MODEL $READ_DIR > to_map.fa


# 2. Align reads to reference

# align.py calls BWA to align the basecalls to the reference
$SLOIKA_ROOT/misc/align.py $GENOME to_map.fa
# This command extracts a reference sequence for each read using coordinates from the SAM file.
sloika/misc/refs_from_sam.py --output_strand_list to_map.txt --pad 50 $GENOME to_map.sam > to_map_refs.fa


# 3. Remap reads using existing model
export OMP_NUM_THREADS=1
export THEANO_FLAGS=device=cpu,floatX=float32
sloika/bin/chunkify.py raw_remap --overwrite --quiet --input_strand_list to_map.txt --downsample 5 $READ_DIR batch_remapped.hdf5 $MODEL to_map_refs.fa


# 4. Train a new model

# Comment the following line to train on the CPU.
# You may need to adjust these flags for your machine, GPU, and current system load
# see TODO: link to wiki
#export THEANO_FLAGS=openmp=True,floatX=float32,warn_float64=warn,optimizer=fast_run,device=gpu0,lib.cnmem=0.4
mkdir -p $WORK_DIR/training
TRAIN_DIR=$(mktemp -d $WORK_DIR/training/baseline_raw_gru_XXXXX)
$SLOIKA_ROOT/bin/train_network.py raw --batch 50 --stride 5 --overwrite --niteration 1 sloika/models/baseline_raw_gru.py $TRAIN_DIR batch_remapped.hdf5


# We exit here as the remaining steps are a work in progress
exit 0


# 5. Evaluate new model on test data

# compile newly trained model into scrappie
git clone https://git.oxfordnanolabs.local/algorithm/scrappie.git
(cd scrappie && git checkout raw)
scrappie/misc/parse_gru_raw.py $TRAIN_DIR/model_final.pkl > scrappie/src/nanonet_raw.h
(cd scrappie && mkdir -p test && cd test && cmake .. && make)

# basecall and align to reference
export OMP_NUM_THREADS=$NPROC
export OPENBLAS_NUM_THREADS=1
tail -n +2 test.txt | xargs scrappie/test/scrappie raw > test.fa
sloika/misc/align.py $GENOME test.fa
