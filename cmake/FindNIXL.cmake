# Copyright (c) 2026, NVIDIA CORPORATION.
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
# See the License for the specific language governing permissions
# and limitations under the License.
#

# Locates a NIXL install (headers + shared libs). Search order:
#   1. -DNIXL_ROOT=... or $NIXL_ROOT
#   2. /home/aocsa/git/nixl/install  (source-tree prefix used on this box)
#   3. /opt/nvidia/nvda_nixl         (NIXL meson default prefix)
#
# Result variables: NIXL_FOUND, NIXL_INCLUDE_DIRS, NIXL_LIBRARIES, NIXL_LIBRARY_DIR

if(NOT NIXL_ROOT AND DEFINED ENV{NIXL_ROOT})
  set(NIXL_ROOT "$ENV{NIXL_ROOT}")
endif()

set(_NIXL_HINTS)
if(NIXL_ROOT)
  list(APPEND _NIXL_HINTS "${NIXL_ROOT}")
endif()
list(APPEND _NIXL_HINTS
  "/home/aocsa/git/nixl/install"
  "/opt/nvidia/nvda_nixl")

find_path(NIXL_INCLUDE_DIR
  NAMES nixl.h
  HINTS ${_NIXL_HINTS}
  PATH_SUFFIXES include)

find_library(NIXL_LIBRARY
  NAMES nixl
  HINTS ${_NIXL_HINTS}
  PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu)

find_library(NIXL_BUILD_LIBRARY
  NAMES nixl_build
  HINTS ${_NIXL_HINTS}
  PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu)

find_library(NIXL_COMMON_LIBRARY
  NAMES nixl_common
  HINTS ${_NIXL_HINTS}
  PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu)

find_library(NIXL_SERDES_LIBRARY
  NAMES serdes
  HINTS ${_NIXL_HINTS}
  PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(NIXL
  DEFAULT_MSG
  NIXL_LIBRARY
  NIXL_BUILD_LIBRARY
  NIXL_COMMON_LIBRARY
  NIXL_SERDES_LIBRARY
  NIXL_INCLUDE_DIR)

if(NIXL_FOUND)
  set(NIXL_INCLUDE_DIRS ${NIXL_INCLUDE_DIR})
  set(NIXL_LIBRARIES
    ${NIXL_LIBRARY}
    ${NIXL_BUILD_LIBRARY}
    ${NIXL_COMMON_LIBRARY}
    ${NIXL_SERDES_LIBRARY})
  get_filename_component(NIXL_LIBRARY_DIR ${NIXL_LIBRARY} DIRECTORY)
  mark_as_advanced(
    NIXL_INCLUDE_DIR
    NIXL_LIBRARY
    NIXL_BUILD_LIBRARY
    NIXL_COMMON_LIBRARY
    NIXL_SERDES_LIBRARY)
endif()
