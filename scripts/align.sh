#!/usr/bin/env bash
FILES=$1
GENOME=/mnt/data/human/reference/g1kv37.fa.gz
SLOIKA=/home/OXFORDNANOLABS/tmassingham/git/sloika
TANG=/home/OXFORDNANOLABS/tmassingham/git/tang
CHIMAERA=/home/OXFORDNANOLABS/tmassingham/git/chimaera

TANGBIN=${TANG}/bin
MISMATCH=${CHIMAERA}/data/model/mismatch_scores.txt

for FA in $@
do
	PREF=${FA%%.fa}
	stdbuf -o 0 -e 0 -i 0 bwa mem -x ont2d -t 32 -A 1 -B 2 -O 2 -E 1 ${GENOME} ${FA} > ${PREF}.sam 
	${SLOIKA}/misc/samacc.py ${PREF}.sam > ${PREF}.samacc
	R --slave --vanilla -e "
pref = commandArgs(trailingOnly=T)[1]; \
a = read.table(paste(pref, '.samacc', sep=''), h=T); \
da = density(a\$accuracy); \
print(length(unique(a\$name2))); \
print(mean(a\$accuracy)); \
print(da\$x[which.max(da\$y)]); \
print(quantile(a\$accuracy, c(5, 25, 50, 75, 95) / 100)); \
print(sum(a\$accuracy >= 0.9) / nrow(a)); \
png(paste(pref, '.png', sep='')); \
hist(100 * a\$accuracy, nclass=50, col='cornflowerblue', xlab='Accuracy', main=pref, xlim=c(65,100)); \
abline(v=100 * da\$x[which.max(da\$y)], col='coral4', lty='dashed', lwd=2); \
grid(); \
dev.off();
" --args ${PREF} > ${PREF}.summary
<<"COMMENT"
	${TANGBIN}/align_fasta.py ${MISMATCH} ${GENOME} ${FA} > ${PREF}.lastal
        R --slave --vanilla -e "
pref = commandArgs(trailingOnly=T)[1]; \
a = read.table(paste(pref, '.lastal', sep=''), h=T); \
da = density(a\$accuracy); \
print(length(unique(a\$name2))); \
print(mean(a\$accuracy)); \
print(da\$x[which.max(da\$y)]); \
print(quantile(a\$accuracy, c(5, 25, 50, 75, 95) / 100)); \
print(sum(a\$accuracy >= 0.9) / nrow(a)); \
png(paste(pref, '.png', sep='')); \
hist(100 * a\$accuracy, nclass=50, col='cornflowerblue', xlab='Accuracy', main=pref, xlim=c(75,100)); \
abline(v=100 * da\$x[which.max(da\$y)], col='coral4', lty='dashed', lwd=2); \
grid(); \
dev.off();
" --args ${PREF} > ${PREF}.summary2
COMMENT

done
