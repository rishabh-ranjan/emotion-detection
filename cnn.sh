#!/bin/bash
### Set the job name (for your reference)
#PBS -N ccnn_0.001_64 -e ccnn_0.001_64.err -o ccnn_0.001_64.out
### Set the project name, your department code by default
#PBS -P cse
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ncpus=6:ngpus=1:centos=skylake
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=2:00:00

#PBS -l software=replace_with_Your_software_name
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
#job 
source ~/.bashrc
conda activate col774
python3 cnn.py 0.001 64 100
#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
