#!/bin/bash

# Configuration for CPU cluster SSH connection
CPU_CLUSTER_HOST="${CPU_CLUSTER_HOST:-dione.mpi-inf.mpg.de}"  # Set your CPU cluster hostname
CPU_CLUSTER_USER="${CPU_CLUSTER_USER:-kwijaya}"         # Set your SSH username
REMOTE_WORK_DIR="${REMOTE_WORK_DIR:-/HPS/molgen/work/creative-generativeai-diffusion}"  # Remote working directory

cd src/optimization_loop/Inner_loop || exit 1  # exit if directory not found

for ((i=0; i<15; i++)); do
    echo "=== Iteration $i ==="

    if [ "$i" -eq 0 ]; then
        # Run inner_loop_creation_1 (GPU only, no Docker)
        python innerloop_creation_1.py --iteration "$i"
        
        # Run inner_loop_creation_2 (CPU + Docker) via SSH to CPU cluster
        echo "Running innerloop_creation_2.py on CPU cluster via SSH..."
        ssh "${CPU_CLUSTER_USER}@${CPU_CLUSTER_HOST}" "cd ${REMOTE_WORK_DIR}/src/optimization_loop/Inner_loop && conda activate creative-genai && python3 innerloop_creation_2.py --iteration $i"
        
        if [ $? -ne 0 ]; then
            echo "Error: innerloop_creation_2.py failed on CPU cluster"
            exit 1
        fi
    fi
    # Run inner_loop_creation_3 (GPU only, no Docker)
    python innerloop_creation_3.py --iteration "$i"

    # Run inner_loop_creation_4 (CPU + Docker) via SSH to CPU cluster

    echo "Running innerloop_creation_4.py on CPU cluster via SSH..."
    ssh "${CPU_CLUSTER_USER}@${CPU_CLUSTER_HOST}" "cd ${REMOTE_WORK_DIR}/src/optimization_loop/Inner_loop && conda activate creative-genai && python3 innerloop_creation_4.py --iteration $i"
    echo "Iteration $i completed!"
done
