Documentation for the code

How to run it on Sockeye

Once you're in cuslater directory, run the following commands:

	1. cmake .
	2. make simple
	3. Request a gpu node: 
		```srun --account=st-greif-1-gpu --partition=interactive_gpu --time=3:0:0 -N 1 -n 2 --mem=32G --gpus=1 --cpus-per-task 1 --pty /bin/bash```
	4. ./simple
	

Other comments by Gautam: all of the relevant code is in `src/evalIntegral.cu`
