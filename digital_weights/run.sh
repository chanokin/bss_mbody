#!/bin/bash
# module load nmpm_software/current
module load nmpm_software/2019-04-23-4189_5834-1

srun -p experiment --wmod 33 singularity exec --app visionary-wafer /containers/stable/latest \
python digital_weights.py
