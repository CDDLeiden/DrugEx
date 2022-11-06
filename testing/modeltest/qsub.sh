#!/bin/bash
#PBS -N drugex_testing
#PBS -q cheminf
#PBS -l select=1:ncpus=12:ngpus=3:mem=16gb
#PBS -l walltime=720:00:00
#PBS -m ae

set -e

# set important variables
export MODEL=${MODEL}
export CONDA_ENV=${CONDA_ENV}
export EXPERIMENT_ID=${EXPERIMENT_ID}
export WORKDIR=${WORKDIR:-`pwd`}
export N_EPOCHS=${N_EPOCHS:-30}

[ -z "$MODEL" ] && echo "\$MODEL is empty. Exiting..." && exit 1
[ -z "$CONDA_ENV" ] && echo "\$CONDA_ENV is empty. Exiting..." && exit 1
[ -z "$EXPERIMENT_ID" ] && echo "\$EXPERIMENT_ID is empty. Exiting..." && exit 1
[ -z "$WORKDIR" ] && echo "\$WORKDIR is empty. Exiting..." && exit 1

export EXPERIMENT_ID="${MODEL}_${EXPERIMENT_ID}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export N_EPOCHS="${N_EPOCHS:-30}"

# work begins here
export SCRATCHDIR=/scratch/$USER/$PBS_JOBID # this might be useful -> we can get this variable from python and fetch big data there
mkdir $SCRATCHDIR
export OUTDIR=$WORKDIR/outputs
mkdir -p $OUTDIR

# append a line to a file "jobs_info.txt" containing the ID of the job and the current worker hostname
echo "$PBS_JOBID is running on node `hostname -f`." >> $WORKDIR/jobs_info.txt

# go to the working directory and copy over files
cp -r $WORKDIR/*.py $SCRATCHDIR/
cp -r $WORKDIR/$MODEL $SCRATCHDIR/
cp -r $WORKDIR/data $SCRATCHDIR/data
cd $SCRATCHDIR

# activate the conda environment
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate $CONDA_ENV

export PYTHONPATH=`pwd`:$PYTHONPATH
python run.py && cp -TR $SCRATCHDIR/output $OUTDIR/output_${EXPERIMENT_ID} && rm -rf $SCRATCHDIR
