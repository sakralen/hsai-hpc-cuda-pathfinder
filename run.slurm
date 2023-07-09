#!/bin/bash
#SBATCH -J pf
#SBATCH -N 1
#SBATCH -p tornado-k40
#SBATCH -o ./out/std/pf-%j.out
#SBATCH -e ./out/err/pf-%j.err
#SBATCH -t 00:10:00 

if [[ -f /etc/profile.d/modules-basis.sh ]]; then
source /etc/profile.d/modules-basis.sh
fi

module load nvidia/cuda/10.1

./build/bin/pathfinder %1 %2 %3


