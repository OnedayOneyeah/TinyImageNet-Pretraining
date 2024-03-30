#!/bin/bash

# GPU=$1
GPU=7
CPU="$((GPU * 8))-$((GPU * 8 + 7))"

echo "!! GPU ${GPU}, CPU ${CPU} !!"

# taskset -c ${CPU} python main.py --gpu_id 6,7
# taskset -c ${CPU} python main.py --gpu_id 3,5,6,7 --no_jsd
# taskset -c ${CPU} python main.py --gpu_id  4,5 --alpha_trans
taskset -c ${CPU} python main.py --gpu_id  6,7
# taskset -c ${CPU} python test.py --gpu_id  4,5