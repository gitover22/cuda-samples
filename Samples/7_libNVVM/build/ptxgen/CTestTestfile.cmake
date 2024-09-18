# CMake generated Testfile for 
# Source directory: /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/ptxgen
# Build directory: /raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build/ptxgen
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ptxgenTest "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build/ptxgen/ptxgen" "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/ptxgen/test.ll")
set_tests_properties(ptxgenTest PROPERTIES  FIXTURES_SETUP "PTXGENTEST" WORKING_DIRECTORY "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/build" _BACKTRACE_TRIPLES "/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/ptxgen/CMakeLists.txt;30;add_test;/raid/home/zouguoqiang/cuda-samples/Samples/7_libNVVM/ptxgen/CMakeLists.txt;0;")
