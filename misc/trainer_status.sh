#!/usr/bin/env bash
nvidia-smi

TRAINDIR=${HOME}/trainer
echo
echo 'Jobs running'
echo '============'
echo
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
echo
(
	echo -e "Index\tModel\tTrainingData\tOutputDirectory"
	echo -e "-----\t-----\t------------\t---------------"
	echo
	sqlite3 ${TRAINDIR}/runs.db 'select model, training_data, output_directory from runs where status = 3' | tr '|' '\t' | cat -n
) | column -t 
echo

echo
echo 'Jobs pending'
echo '============'
echo
(	echo -e "RunID\tModel\tPriority\tTrainingData\tOutputDirectory"
	echo -e "-----\t-----\t--------\t------------\t---------------"
	echo
	sqlite3 ${TRAINDIR}/runs.db 'select runid, model, priority, training_data, output_directory from runs where status = 0 order by priority' | tr '|' '\t'
) | column -t
echo


echo
echo 'Jobs suspended'
echo '=============='
echo
(
	echo -e "Index\tModel\tTrainingData\tOutputDirectory"
	echo -e "-----\t-----\t------------\t---------------"
	echo
	sqlite3 ${TRAINDIR}/runs.db 'select model, training_data, output_directory from runs where status = 4' | tr '|' '\t' | cat -n
) | column -t
echo
