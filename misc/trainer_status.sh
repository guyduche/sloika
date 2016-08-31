#!/usr/bin/env bash
nvidia-smi

TRAINDIR=${HOME}/trainer
echo
echo 'Jobs running'
echo '============'
JOBS=`sqlite3 ${TRAINDIR}/runs.db 'select output_directory from runs where status = 1' | tr '|' '\t'`
for JOB in ${JOBS}
do
    echo '> ' `basename ${JOB}`
    tail -n 2 ${TRAINDIR}/${JOB}/model.log | head -n 1
done
echo

echo
echo 'Jobs failed'
echo '==========='
sqlite3 ${TRAINDIR}/runs.db 'select model, training_data, output_directory from runs where status = 3' | tr '|' '\t' | column -t | cat -n
echo

echo
echo 'Jobs pending'
echo '============'
sqlite3 ${TRAINDIR}/runs.db 'select model, training_data, output_directory from runs where status = 0 order by priority' | tr '|' '\t' | column -t | cat -n
echo


echo
echo 'Jobs suspended'
echo '=============='
sqlite3 ${TRAINDIR}/runs.db 'select model, training_data, output_directory from runs where status = 4' | tr '|' '\t' | column -t | cat -n
echo
