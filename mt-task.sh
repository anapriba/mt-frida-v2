#!/bin/bash

#PBS -N hello
#PBS -l select=1:ncpus=16:mem=64GB:ngpus=1

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

# ls projekt/mt/

echo "----- Running code ------ "

# apptainer exec projekt/mt/mt.sif python3 projekt/mt/main.py --fix-train=True

apptainer exec projekt/mt/mt.sif python3 projekt/mt/data.py

# apptainer exec projekt/mt/mt.sif python3 projekt/mt/train.py

