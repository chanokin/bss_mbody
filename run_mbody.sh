#!/bin/bash
# module load nmpm_software/current
module load nmpm_software/2019-04-23-4189_5834-1

srun -p experiment --wmod 30 run_nmpm_software \
python bss-mbody.py 50 500 0 50 0 0 --nSamplesAL 2
