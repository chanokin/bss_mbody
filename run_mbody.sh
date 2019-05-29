#!/bin/bash
# module load nmpm_software/current
module load nmpm_software/2019-04-23-4189_5834-1

srun -p experiment --wmod 30 singularity exec --app visionary-wafer /containers/stable/latest \
python bss-mbody.py 20 250 0 20 0 0 --nSamplesAL 2
