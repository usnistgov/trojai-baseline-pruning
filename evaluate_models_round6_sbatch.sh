#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --job-name=base_trojai
#SBATCH -o log-%N.%j.out
#SBATCH --time=48:0:0
printf -v numStr "%08d" ${1}
echo "id-$numStr"
source /apps/anaconda3/etc/profile.d/conda.sh
conda activate r5venv

NUM_SAMPLES=$1
NUM_IMAGES=$2
PRUNING_METHOD=$3

#CONTAINER_NAME=$1
#QUEUE_NAME=$2
#MODEL_DIR=$3

#MODEL_DIR=/home/pnb/raid1/trojai/datasets/round6/round6-train-dataset
MODEL_DIR=/wrk/pnb/trojai_data/round6/round6-train-dataset/models
EMBEDDING_DIRPATH=/wrk/pnb/trojai_data/round6/round6-train-dataset/embeddings
TOKENIZER_DIRPATH=/wrk/pnb/trojai_data/round6/round6-train-dataset/tokenizers

#ACTIVE_DIR=/home/trojai/active

#CONTAINER_EXEC=/mnt/scratch/$CONTAINER_NAME
#RESULT_DIR=/home/pnb/raid1/trojai/datasets/round6/scratch_r6
#SCRATCH_DIR=/home/pnb/raid1/trojai/datasets/round6/scratch_r6
RESULT_DIR=/wrk/pnb/trojai_data/round6/scratch_r6-nS$NUM_SAMPLES-nD$NUM_IMAGES-$PRUNING_METHOD
SCRATCH_DIR=/wrk/pnb/trojai_data/round6/scratch_r6-nS$NUM_SAMPLES-nD$NUM_IMAGES-$PRUNING_METHOD

#mkdir -p $RESULT_DIR
mkdir -p $SCRATCH_DIR

# only use the name obfuscation outside the STS queue
#if [[ "$QUEUE_NAME" != "sts" ]]; then
#	mkdir -p $ACTIVE_DIR
#fi


# find all the 'id-' model files and shuffle their iteration order
for dir in `find $MODEL_DIR -maxdepth 1 -type d | shuf`
do
	# check that the directory is not the root MODEL_DIR
	if [ "$dir" != "$MODEL_DIR" ]; then
		# check that the directory starts with "id"
		MODEL="$(basename $dir)"

		if [[ $MODEL == id* ]] ; then

			python ./trojan_detector_nlp.py --model_filepath $dir/model.pt  --result_filepath $RESULT_DIR/test_python_output.txt --scratch_dirpath $SCRATCH_DIR --examples_dirpath $dir/clean_example_data  --num_samples $NUM_SAMPLES --num_images_used $NUM_IMAGES --pruning_method $PRUNING_METHOD --embedding_dirpath $EMBEDDING_DIRPATH --tokenizer_dirpath $TOKENIZER_DIRPATH
			echo "Finished executing $dir, returned status code: $?"

#			if [[ "$QUEUE_NAME" == "sts" ]]; then
#				python ./trojan_detector_round2.py --model_filepath $dir/model.pt  --result_filepath $RESULT_DIR/test_python_output.txt --scratch_dirpath $SCRATCH_DIR --examples_dirpath $dir/example_data
#				#singularity run --contain -B /mnt/scratch -B /home/trojai/data --nv $CONTAINER_EXEC --model_filepath $dir/model.pt --result_filepath $RESULT_DIR/$MODEL.txt --scratch_dirpath $SCRATCH_DIR --examples_dirpath $dir/example_data
#				echo "Finished executing $dir, returned status code: $?"
#			else
#				# pre-preemptively clean up the active directory
#				rm -rf $ACTIVE_DIR/*
#				# Copy model to the active folder to obscure its name
#				cp -r $dir/* $ACTIVE_DIR
#
#				singularity run --contain -B /mnt/scratch -B $ACTIVE_DIR --nv $CONTAINER_EXEC --model_filepath $ACTIVE_DIR/model.pt --result_filepath $ACTIVE_DIR/result.txt --scratch_dirpath $SCRATCH_DIR --examples_dirpath $ACTIVE_DIR/example_data >> $RESULT_DIR/$CONTAINER_NAME.out 2>&1
#				echo "Finished executing, returned status code: $?" >> $RESULT_DIR/$CONTAINER_NAME.out 2>&1
#
#				if [[ -f $ACTIVE_DIR/result.txt ]]; then
#					# copy result back to real output filename based on model name
#					cp $ACTIVE_DIR/result.txt  $RESULT_DIR/$MODEL.txt
#				fi
#			fi
		fi
	fi
done
