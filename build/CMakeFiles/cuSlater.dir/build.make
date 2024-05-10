# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cmake-3.26.3-jymy3coxptnuxi2nos4tpenemitu7hfr/bin/cmake

# The command to remove a file.
RM = /arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cmake-3.26.3-jymy3coxptnuxi2nos4tpenemitu7hfr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /project/st-greif-1/mewert/cuslater

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /project/st-greif-1/mewert/cuslater/build

# Include any dependencies generated for this target.
include CMakeFiles/cuSlater.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cuSlater.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cuSlater.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuSlater.dir/flags.make

CMakeFiles/cuSlater.dir/src/gputensors.cu.o: CMakeFiles/cuSlater.dir/flags.make
CMakeFiles/cuSlater.dir/src/gputensors.cu.o: CMakeFiles/cuSlater.dir/includes_CUDA.rsp
CMakeFiles/cuSlater.dir/src/gputensors.cu.o: /project/st-greif-1/mewert/cuslater/src/gputensors.cu
CMakeFiles/cuSlater.dir/src/gputensors.cu.o: CMakeFiles/cuSlater.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/project/st-greif-1/mewert/cuslater/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/cuSlater.dir/src/gputensors.cu.o"
	/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuSlater.dir/src/gputensors.cu.o -MF CMakeFiles/cuSlater.dir/src/gputensors.cu.o.d -x cu -rdc=true -c /project/st-greif-1/mewert/cuslater/src/gputensors.cu -o CMakeFiles/cuSlater.dir/src/gputensors.cu.o

CMakeFiles/cuSlater.dir/src/gputensors.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuSlater.dir/src/gputensors.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuSlater.dir/src/gputensors.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuSlater.dir/src/gputensors.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cuSlater.dir/src/svalues.cu.o: CMakeFiles/cuSlater.dir/flags.make
CMakeFiles/cuSlater.dir/src/svalues.cu.o: CMakeFiles/cuSlater.dir/includes_CUDA.rsp
CMakeFiles/cuSlater.dir/src/svalues.cu.o: /project/st-greif-1/mewert/cuslater/src/svalues.cu
CMakeFiles/cuSlater.dir/src/svalues.cu.o: CMakeFiles/cuSlater.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/project/st-greif-1/mewert/cuslater/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/cuSlater.dir/src/svalues.cu.o"
	/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuSlater.dir/src/svalues.cu.o -MF CMakeFiles/cuSlater.dir/src/svalues.cu.o.d -x cu -rdc=true -c /project/st-greif-1/mewert/cuslater/src/svalues.cu -o CMakeFiles/cuSlater.dir/src/svalues.cu.o

CMakeFiles/cuSlater.dir/src/svalues.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuSlater.dir/src/svalues.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuSlater.dir/src/svalues.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuSlater.dir/src/svalues.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cuSlater.dir/src/stovalues.cu.o: CMakeFiles/cuSlater.dir/flags.make
CMakeFiles/cuSlater.dir/src/stovalues.cu.o: CMakeFiles/cuSlater.dir/includes_CUDA.rsp
CMakeFiles/cuSlater.dir/src/stovalues.cu.o: /project/st-greif-1/mewert/cuslater/src/stovalues.cu
CMakeFiles/cuSlater.dir/src/stovalues.cu.o: CMakeFiles/cuSlater.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/project/st-greif-1/mewert/cuslater/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/cuSlater.dir/src/stovalues.cu.o"
	/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuSlater.dir/src/stovalues.cu.o -MF CMakeFiles/cuSlater.dir/src/stovalues.cu.o.d -x cu -rdc=true -c /project/st-greif-1/mewert/cuslater/src/stovalues.cu -o CMakeFiles/cuSlater.dir/src/stovalues.cu.o

CMakeFiles/cuSlater.dir/src/stovalues.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuSlater.dir/src/stovalues.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuSlater.dir/src/stovalues.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuSlater.dir/src/stovalues.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cuSlater.dir/src/utilities.cu.o: CMakeFiles/cuSlater.dir/flags.make
CMakeFiles/cuSlater.dir/src/utilities.cu.o: CMakeFiles/cuSlater.dir/includes_CUDA.rsp
CMakeFiles/cuSlater.dir/src/utilities.cu.o: /project/st-greif-1/mewert/cuslater/src/utilities.cu
CMakeFiles/cuSlater.dir/src/utilities.cu.o: CMakeFiles/cuSlater.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/project/st-greif-1/mewert/cuslater/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/cuSlater.dir/src/utilities.cu.o"
	/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuSlater.dir/src/utilities.cu.o -MF CMakeFiles/cuSlater.dir/src/utilities.cu.o.d -x cu -rdc=true -c /project/st-greif-1/mewert/cuslater/src/utilities.cu -o CMakeFiles/cuSlater.dir/src/utilities.cu.o

CMakeFiles/cuSlater.dir/src/utilities.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuSlater.dir/src/utilities.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuSlater.dir/src/utilities.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuSlater.dir/src/utilities.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cuSlater.dir/src/grids.cu.o: CMakeFiles/cuSlater.dir/flags.make
CMakeFiles/cuSlater.dir/src/grids.cu.o: CMakeFiles/cuSlater.dir/includes_CUDA.rsp
CMakeFiles/cuSlater.dir/src/grids.cu.o: /project/st-greif-1/mewert/cuslater/src/grids.cu
CMakeFiles/cuSlater.dir/src/grids.cu.o: CMakeFiles/cuSlater.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/project/st-greif-1/mewert/cuslater/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/cuSlater.dir/src/grids.cu.o"
	/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuSlater.dir/src/grids.cu.o -MF CMakeFiles/cuSlater.dir/src/grids.cu.o.d -x cu -rdc=true -c /project/st-greif-1/mewert/cuslater/src/grids.cu -o CMakeFiles/cuSlater.dir/src/grids.cu.o

CMakeFiles/cuSlater.dir/src/grids.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuSlater.dir/src/grids.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuSlater.dir/src/grids.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuSlater.dir/src/grids.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o: CMakeFiles/cuSlater.dir/flags.make
CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o: CMakeFiles/cuSlater.dir/includes_CUDA.rsp
CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o: /project/st-greif-1/mewert/cuslater/src/evalIntegral.cu
CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o: CMakeFiles/cuSlater.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/project/st-greif-1/mewert/cuslater/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o"
	/arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o -MF CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o.d -x cu -rdc=true -c /project/st-greif-1/mewert/cuslater/src/evalIntegral.cu -o CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o

CMakeFiles/cuSlater.dir/src/evalIntegral.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuSlater.dir/src/evalIntegral.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuSlater.dir/src/evalIntegral.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuSlater.dir/src/evalIntegral.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cuSlater
cuSlater_OBJECTS = \
"CMakeFiles/cuSlater.dir/src/gputensors.cu.o" \
"CMakeFiles/cuSlater.dir/src/svalues.cu.o" \
"CMakeFiles/cuSlater.dir/src/stovalues.cu.o" \
"CMakeFiles/cuSlater.dir/src/utilities.cu.o" \
"CMakeFiles/cuSlater.dir/src/grids.cu.o" \
"CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o"

# External object files for target cuSlater
cuSlater_EXTERNAL_OBJECTS =

CMakeFiles/cuSlater.dir/cmake_device_link.o: CMakeFiles/cuSlater.dir/src/gputensors.cu.o
CMakeFiles/cuSlater.dir/cmake_device_link.o: CMakeFiles/cuSlater.dir/src/svalues.cu.o
CMakeFiles/cuSlater.dir/cmake_device_link.o: CMakeFiles/cuSlater.dir/src/stovalues.cu.o
CMakeFiles/cuSlater.dir/cmake_device_link.o: CMakeFiles/cuSlater.dir/src/utilities.cu.o
CMakeFiles/cuSlater.dir/cmake_device_link.o: CMakeFiles/cuSlater.dir/src/grids.cu.o
CMakeFiles/cuSlater.dir/cmake_device_link.o: CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o
CMakeFiles/cuSlater.dir/cmake_device_link.o: CMakeFiles/cuSlater.dir/build.make
CMakeFiles/cuSlater.dir/cmake_device_link.o: /arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/lib64/libcublasLt.so
CMakeFiles/cuSlater.dir/cmake_device_link.o: /arc/project/st-greif-1/sfw/libcutensor-linux-x86_64-1.6.2.3-archive/lib/11.0/libcutensor.so
CMakeFiles/cuSlater.dir/cmake_device_link.o: CMakeFiles/cuSlater.dir/deviceLinkLibs.rsp
CMakeFiles/cuSlater.dir/cmake_device_link.o: CMakeFiles/cuSlater.dir/deviceObjects1.rsp
CMakeFiles/cuSlater.dir/cmake_device_link.o: CMakeFiles/cuSlater.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/project/st-greif-1/mewert/cuslater/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CUDA device code CMakeFiles/cuSlater.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuSlater.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuSlater.dir/build: CMakeFiles/cuSlater.dir/cmake_device_link.o
.PHONY : CMakeFiles/cuSlater.dir/build

# Object files for target cuSlater
cuSlater_OBJECTS = \
"CMakeFiles/cuSlater.dir/src/gputensors.cu.o" \
"CMakeFiles/cuSlater.dir/src/svalues.cu.o" \
"CMakeFiles/cuSlater.dir/src/stovalues.cu.o" \
"CMakeFiles/cuSlater.dir/src/utilities.cu.o" \
"CMakeFiles/cuSlater.dir/src/grids.cu.o" \
"CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o"

# External object files for target cuSlater
cuSlater_EXTERNAL_OBJECTS =

libcuSlater.so: CMakeFiles/cuSlater.dir/src/gputensors.cu.o
libcuSlater.so: CMakeFiles/cuSlater.dir/src/svalues.cu.o
libcuSlater.so: CMakeFiles/cuSlater.dir/src/stovalues.cu.o
libcuSlater.so: CMakeFiles/cuSlater.dir/src/utilities.cu.o
libcuSlater.so: CMakeFiles/cuSlater.dir/src/grids.cu.o
libcuSlater.so: CMakeFiles/cuSlater.dir/src/evalIntegral.cu.o
libcuSlater.so: CMakeFiles/cuSlater.dir/build.make
libcuSlater.so: /arc/software/spack-2023/opt/spack/linux-centos7-skylake_avx512/gcc-9.4.0/cuda-11.3.1-kupglxsfujxxyigzlszl6tdpmy7igxop/lib64/libcublasLt.so
libcuSlater.so: /arc/project/st-greif-1/sfw/libcutensor-linux-x86_64-1.6.2.3-archive/lib/11.0/libcutensor.so
libcuSlater.so: CMakeFiles/cuSlater.dir/cmake_device_link.o
libcuSlater.so: CMakeFiles/cuSlater.dir/linkLibs.rsp
libcuSlater.so: CMakeFiles/cuSlater.dir/objects1.rsp
libcuSlater.so: CMakeFiles/cuSlater.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/project/st-greif-1/mewert/cuslater/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CUDA shared library libcuSlater.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuSlater.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuSlater.dir/build: libcuSlater.so
.PHONY : CMakeFiles/cuSlater.dir/build

CMakeFiles/cuSlater.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuSlater.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuSlater.dir/clean

CMakeFiles/cuSlater.dir/depend:
	cd /project/st-greif-1/mewert/cuslater/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /project/st-greif-1/mewert/cuslater /project/st-greif-1/mewert/cuslater /project/st-greif-1/mewert/cuslater/build /project/st-greif-1/mewert/cuslater/build /project/st-greif-1/mewert/cuslater/build/CMakeFiles/cuSlater.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuSlater.dir/depend

