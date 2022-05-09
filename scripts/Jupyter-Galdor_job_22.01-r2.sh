#!/bin/bash
#PBS -N JupyterLabJob
#PBS -q gpu_long
#PBS -l select=1:ncpus=12:mem=32gb:scratch_local=50gb:ngpus=1:cluster=galdor
#PBS -l walltime=336:00:00
#PBS -m ae
# The 4 lines above are options for scheduling system: job will run 1 hour at maximum, 1 machine with 4 processors + 4gb RAM memory + 10gb scratch memory are requested, email notification will be sent when the job aborts (a) or ends (e)

echo ${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}

# define variables
SING_IMAGE="/storage/plzen4-ntis/projects/cv/CarnivoreID/carnivore_id_v1.2.sif"
HOMEDIR=/storage/plzen4-ntis/home/$USER # substitute username and path to to your real username and path
HOSTNAME=`hostname -f`
JUPYTER_PORT="8888"
IMAGE_BASE=`basename $SING_IMAGE`
export PYTHONUSERBASE=$HOMEDIR/.local-${IMAGE_BASE}

mkdir -p ${PYTHONUSERBASE}/lib/python3.8/site-packages 

#find nearest free port to listen
isfree=$(netstat -taln | grep $JUPYTER_PORT)
while [[ -n "$isfree" ]]; do
    JUPYTER_PORT=$[JUPYTER_PORT+1]
    isfree=$(netstat -taln | grep $JUPYTER_PORT)
done


# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

#set SINGULARITY variables for runtime data
export SINGULARITY_CACHEDIR=$HOMEDIR
export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR
export SINGULARITY_TMPDIR=$SCRATCHDIR
export SINGULARITYENV_PREPEND_PATH=$PYTHONUSERBASE/bin:$PATH


# move into $HOME directory
cd $HOMEDIR 
if [ ! -f ./.jupyter/jupyter_notebook_config.json ]; then
   echo "jupyter passwd reset!"
   mkdir -p .jupyter/
   #here you can commem=nt randomly generated password and set your password
   pass=`dd if=/dev/urandom count=1 2> /dev/null | uuencode -m - | sed -ne 2p | cut -c-12` ; echo $pass
   #pass="SecretPassWord" 
   hash=`singularity exec $SING_IMAGE python -c "from notebook.auth import passwd ; hash = passwd('$pass') ; print(hash)" 2>/dev/null`
   cat > .jupyter/jupyter_notebook_config.json << EOJson
{
  "NotebookApp": {
      "password": "$hash"
    }
}
EOJson
  PASS_MESSAGE="Your password was set to '$pass' (without ticks)."
else
  PASS_MESSAGE="Your password was already set before."
fi

# MAIL to user HOSTNAME
# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
#echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

EXECMAIL=`which mail`
$EXECMAIL -s "JupyterLab job is running on $HOSTNAME:$JUPYTER_PORT" $PBS_O_LOGNAME << EOFmail
Job with JupiterLab was started.

Use URL  http://$HOSTNAME:$JUPYTER_PORT 

$PASS_MESSAGE

You can reset password by deleting file .jupyter/jupyter_notebook_config.json and run job again with this script.
EOFmail

# TenzorFlow20.30.simg  is prepared singularity image form NGC, prepare your own :-)

BIND=""
if [[ -d /auto ]]; then
  BIND="$BIND --bind /auto"
fi
if [[ -d $SCRATCHDIR ]]; then
  BIND="$BIND --bind $SCRATCHDIR"
fi
if [[ -d /storage ]]; then
  BIND="$BIND --bind /storage"
fi

singularity exec --nv -H $HOMEDIR \
		 $BIND \
                 $SING_IMAGE jupyter-lab --port $JUPYTER_PORT \

# clean the SCRATCH directory
exec /software/meta-utils/public/clean_scratch
