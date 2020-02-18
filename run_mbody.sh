#!/bin/bash
module load nmpm_software/current
# module load nmpm_software/2019-06-18-4189_5834-1
# module load nmpm_software/2019-04-23-4189_5834-1
# module load nmpm_software/2019-08-14-2
# module load nmpm_software/2019-08-15-5834-1

# HICANNS=205,52,15,77,135,80,131,172,177,8,18,106,2,245,79,168,132,31,\
# 213,3,33,55,249,109,27,37,28,211,176,6,207,139,212,56,17,136,242,14,\
# 244,19,30,57,7,5,16

# HICANNS=76,77,78,79,80,81,74,71,72,73,75,104,105,106,107,108,109,\
# 102,99,100,101,103,136,137,138,139,140,141,134,131,132,133,135,172,\
# 173,174,175,176,177,170,167,168,169,171,208,209,210,211,212,213,206,\
# 203,204,205,207,244,245,246,247,248,249,242,239,240,241,243,32,33,34,\
# 35,36,37,30,27,28,29,31,4,5,6,7,8,9,2,0,1,3,16,17,18,19,20,21,14,12,\
# 13,15,52,53,54,55,56,57,50,47,48,49,51

SELECT=0

HICANNS=0,2,4,5,6,7,8,9,14,15,17,\
18,19,20,21,27,28,29,30,31,32,33,34,35,\
36,37,47,48,49,50,52,53,54,55,56,57,\
71,72,73,74,75,76,77,78,79,80,81,99,100,\
101,102,103,105,106,107,108,109,132,133,134,135,\
136,137,138,139,140,141,167,168,169,170,171,172,173,\
174,175,176,177,203,204,205,206,207,208,209,210,211,\
212,213,239,240,241,242,243,244,245,246,247,248,249

WAFER=33

SAMPLES=100
IN=60
MID=600
OUT=60

# run_nmpm_software
# srun -t 0-3:00 -p experiment --wafer=$WAFER  --hicann=$HICANN \
# singularity exec --app \
# visionary-wafer \
# /containers/stable/2019-04-16_1.img \
# python bss-mbody.py 50 500 0 50 0 0 $WAFER --nSamplesAL 10

if [ $SELECT -eq 1 ]; then
    echo "running with hicanns selected"

    srun -t 0-6:00 -p experiment --wafer=$WAFER  --hicann=$HICANN \
    run_nmpm_software \
    python bss-mbody.py $IN $MID 0 $OUT 0 0 $WAFER --nSamplesAL $SAMPLES
else
    echo "running WITHOUT hicanns selected"

    srun -t 0-6:00 -p experiment --wafer=$WAFER \
    run_nmpm_software \
    python bss-mbody.py $IN $MID 0 $OUT 0 0 $WAFER --nSamplesAL $SAMPLES 
fi
