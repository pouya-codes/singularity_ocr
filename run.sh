#!/bin/bash
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=pouya.ahmadvand@gmail.com     # Where to send mail
#SBATCH --time=4-90:00:00
#SBATCH --output /home/poahmadvand/share/PAN/out.out
#SBATCH --error /home/poahmadvand/share/PAN/err.err

# SBATCH --chdir /projects/ovcare/classification/singularity_modules/singularity_ocr
#SBATCH --workdir /projects/ovcare/classification/singularity_modules/singularity_ocr
#SBATCH --mem=200G

#source /home/poahmadvand/py3env/bin/activate
#python -d "/home/pouya/Develop/UBC/singularity_ocr/Dataset" -o /home/pouya/Develop/UBC/singularity_ocr/Results -l /home/pouya/Develop/UBC/singularity_ocr/Label
singularity run --bind /:/ singularity_ocr.sif -d "/projects/ovcare/classification/pouya/PAN/slides" -o /home/poahmadvand/share/PAN/Results -l /home/poahmadvand/share/PAN/Label