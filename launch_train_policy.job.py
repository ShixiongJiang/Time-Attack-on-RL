#!/bin/bash


#$ -q gpu
#$ -M sjiang5@nd.edu
#$ -m abe
#$ -N train_img_to_attamap
#$ -l gpu_card=4



module load python
python3 train_policy.py
