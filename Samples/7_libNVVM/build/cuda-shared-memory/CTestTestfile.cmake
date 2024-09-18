# CMake generated Testfile for 
# Source directory: /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/cuda-shared-memory
# Build directory: /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build/cuda-shared-memory
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test-cuda-shared-memory-shared_memory "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build/ptxgen/ptxgen" "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/cuda-shared-memory/shared_memory.ll")
set_tests_properties(test-cuda-shared-memory-shared_memory PROPERTIES  FIXTURES_REQUIRED "PTXGENTEST" WORKING_DIRECTORY "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build" _BACKTRACE_TRIPLES "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/cuda-shared-memory/CMakeLists.txt;30;add_test;/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/cuda-shared-memory/CMakeLists.txt;0;")
add_test(test-cuda-shared-memory-extern_shared_memory "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build/ptxgen/ptxgen" "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/cuda-shared-memory/extern_shared_memory.ll")
set_tests_properties(test-cuda-shared-memory-extern_shared_memory PROPERTIES  FIXTURES_REQUIRED "PTXGENTEST" WORKING_DIRECTORY "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build" _BACKTRACE_TRIPLES "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/cuda-shared-memory/CMakeLists.txt;34;add_test;/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/cuda-shared-memory/CMakeLists.txt;0;")
