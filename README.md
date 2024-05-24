# Cuslater
A package for computing four-centered integral calculations in parallel using the CUDA framework.

# Usage for UBC students/faculty/staff
1. Connect to UBC's [sockeye](https://arc.ubc.ca/compute-storage/ubc-arc-sockeye) server
2. Navigate to project direction
3. Run `cmake .` to use the cmakefile to configure the project
4. Run  `make` to compile the test files
5. To run the test files first you will need to get access to the GPUs on the server. This can be done with `srun --account=<ACCOUNT> --partition=interactive_gpu --time=<TIME> -N <NODES> -n 2=<TASKS> --mem=<MEM> --gpus=<GPUS> --cpus-per-task <CPT> --pty /bin/bash` where
  - `<ACCOUNT>` is the specified account the job will be charged with. Example `st-greif-1-gpu`.
  - `<TIME>` is the time limit for the session. Example `1:00:00` would be 1 hour.
  - `<NODES>` is the number of nodes to run.
  - `<TASKS>` is the number of tasks to run.
  - `<MEM>` is the amount of memory to allocate. Example `32G` would be 32 gigabytes.
  - `<GPU>` is how many GPUS to use.
  - `<CPT>` is the number of cpus required per task.
6. Finally you can run the tests. For example, run `./simple` to see the output of the simple test.

