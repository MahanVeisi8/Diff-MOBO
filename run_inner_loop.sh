#!/bin/bash

cd src/optimization_loop/Inner_loop || exit 1  # exit if directory not found

for ((i=0; i<30; i++)); do
    echo "=== Iteration $i ==="

    # Run inner_loop_creation_1 (GPU only, no Docker)
    if [ "$i" -eq 0 ]; then
        python3 innerloop_creation_1.py --iteration "$i"
    fi

    # Run inner_loop_creation_2 (GPU only, no Docker)
    python3 innerloop_creation_2.py --iteration "$i"

    # Run inner_loop_creation_3 (CPU + Docker)
    python3 innerloop_creation_3.py --iteration "$i"

    # Run inner_loop_creation_4 (GPU only, no Docker)
    python3 innerloop_creation_4.py --iteration "$i"

    echo "Iteration $i completed!"
done
