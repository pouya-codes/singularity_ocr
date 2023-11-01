#!/bin/bash
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=pouya.ahmadvand@gmail.com     # Where to send mail
#SBATCH --time=4-90:00:00
#SBATCH --output /projects/ovcare/classification/pouya/Irem/ocr_dov_1.out
#SBATCH --error /projects/ovcare/classification/pouya/Irem/ocr_dov_1.err

# SBATCH --chdir /projects/ovcare/classification/singularity_modules/singularity_ocr
#SBATCH --workdir /projects/ovcare/classification/singularity_modules/singularity_ocr
#SBATCH --mem=150G

#source /home/poahmadvand/py3env/bin/activate
#python -d "/home/pouya/Develop/UBC/singularity_ocr/Dataset" -o /home/pouya/Develop/UBC/singularity_ocr/Results -l /home/pouya/Develop/UBC/singularity_ocr/Label
singularity run --bind /projects:/projects singularity_ocr.sif -d \
"/projects/ovcare/classification/pouya/Irem/globus_mount" -o \
/projects/ovcare/classification/pouya/Irem/DOV/Results_batch_1 -l \
/projects/ovcare/classification/pouya/Irem/DOV/Label_batch_1 \
--num_workers 100
