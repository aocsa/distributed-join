# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.18)
project(distributed LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package(CUDAToolkit REQUIRED)
# CUDA 12.9's imported targets inject targets/*/include as -I, which shadows
# conda CCCL 3.1. Treat those paths as system includes so include/cccl wins.
foreach(_cuda_tgt CUDA::cudart CUDA::cudart_static CUDA::cuda_driver CUDA::nvrtc)
  if(TARGET ${_cuda_tgt})
    get_target_property(_cuda_inc ${_cuda_tgt} INTERFACE_INCLUDE_DIRECTORIES)
    if(_cuda_inc)
      set_property(TARGET ${_cuda_tgt} PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_cuda_inc}")
    endif()
  endif()
endforeach()
unset(_cuda_tgt)
unset(_cuda_inc)
find_package(CUDF REQUIRED)
find_package(rmm CONFIG REQUIRED)
find_package(UCX REQUIRED)
find_package(NCCL REQUIRED)
find_package(MPI REQUIRED)
find_package(NVCOMP REQUIRED)
find_library(RAPIDS_LOGGER_LIBRARY rapids_logger REQUIRED)

# rmm 25.12 needs CCCL's experimental memory_resource (normally pulled in by rmm::rmm).
add_compile_definitions(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

# CUDA 12.2 cannot emit sm_121 (GB10). 12.8+ can emit sm_120, 12.9+ sm_121.
# Override with -DGPU_ARCHS=... if needed.
if(NOT GPU_ARCHS)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9)
    set(GPU_ARCHS "80;90;120;121")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set(GPU_ARCHS "80;90;120")
  else()
    set(GPU_ARCHS "70;80")
  endif()
endif()

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Werror>")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")

add_subdirectory(src lib)
add_subdirectory(benchmark bin/benchmark)
add_subdirectory(test bin/test)
