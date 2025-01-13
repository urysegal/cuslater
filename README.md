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

## Program parameters
Various parameters used in the integral calculation can be configured while running the program like `> ./simple [OPTION] [PARAMETERS...]`
Below is a list of the current configuration options
- --help  Display this help message and exit
- `-a a1 a2 a3 a4`  Set alpha values
- `-c1 x1 y1 z1`  Sets coordinates for c1 (c1.x, c1.y, c1.z)
- `-c2 x2 y2 z2`  Sets coordinates for c2 (c2.x, c2.y, c2.z)
- `-c3 x3 y3 z3`  Sets coordinates for c3 (c3.x, c3.y, c3.z)
- `-c4 x4 y4 z4`  Sets coordinates for c4 (c4.x, c4.y, c4.z)
- `-t tol`  Set tolerance threshold (halts inner loop once value drops below threshold)


The following modules need to be loaded:
module load cuda
module load gcc
module load cmake
module load apptainer

You may need to add the .../gcc-9.4.0/cuda-11.3.1 to CUDA_ROOT and LD_LIBRARY_PATH environment variables.
