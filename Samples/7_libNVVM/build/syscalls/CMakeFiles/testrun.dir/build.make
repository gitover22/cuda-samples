# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build

# Utility rule file for testrun.

# Include the progress variables for this target.
include syscalls/CMakeFiles/testrun.dir/progress.make

testrun: syscalls/CMakeFiles/testrun.dir/build.make

.PHONY : testrun

# Rule to build all files generated by this target.
syscalls/CMakeFiles/testrun.dir/build: testrun

.PHONY : syscalls/CMakeFiles/testrun.dir/build

syscalls/CMakeFiles/testrun.dir/clean:
	cd /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build/syscalls && $(CMAKE_COMMAND) -P CMakeFiles/testrun.dir/cmake_clean.cmake
.PHONY : syscalls/CMakeFiles/testrun.dir/clean

syscalls/CMakeFiles/testrun.dir/depend:
	cd /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/syscalls /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build/syscalls /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build/syscalls/CMakeFiles/testrun.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : syscalls/CMakeFiles/testrun.dir/depend

