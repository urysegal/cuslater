Documentation for the code

## Grids

There are two Julia files for grids. 
	1. `grids.jl` evaluates grids.
			 I have updated grids.jl file to be type flexible. I am storing the grids in `grids_files_adap`
	2. `print_grids.jl` prints grids and saves them in `grids_files_adap` folder.
			Please edit this file and the path in `grids.jl` for calculating new grids.


## Sockeye

Instructions for running on Sockeye.

Once you're in cuslater directory, run the following commands:

	1. cmake .
	2. make simple
	3. Request a gpu node: 
		```srun --account=st-greif-1-gpu --partition=interactive_gpu --time=3:0:0 -N 1 -n 2 --mem=32G --gpus=1 --cpus-per-task 1 --pty /bin/bash```
	4. ./simple
	
## Other Comments
Other comments by Gautam: all of the relevant code is in `src/evalIntegral.cu`
