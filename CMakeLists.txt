cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(Flow360Residual)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)

set(CUDA_MODE 1)

if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH sm_61)
endif(NOT DEFINED CUDA_ARCH)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS};--std=c++14)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=${CUDA_ARCH};-Wno-deprecated-gpu-targets; -lineinfo; -use_fast_math)

set(SOURCES lorenz.cpp plykin_map.cpp plykin_ode.cpp plykin_ode_local.cpp)

set_source_files_properties(
  ${SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

cuda_add_executable(lorenz lorenz.cpp)
target_link_libraries(lorenz ${CUDA_LIBRARIES} png)

cuda_add_executable(plykin_map plykin_map.cpp)
target_link_libraries(plykin_map ${CUDA_LIBRARIES} png)

cuda_add_executable(plykin_ode plykin_ode.cpp)
target_link_libraries(plykin_ode ${CUDA_LIBRARIES} png)

cuda_add_executable(plykin_ode_local plykin_ode_local.cpp)
target_link_libraries(plykin_ode_local ${CUDA_LIBRARIES} png)
