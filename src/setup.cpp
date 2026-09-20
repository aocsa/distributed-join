/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "setup.hpp"

#include "communicator.hpp"
#include "error.hpp"
#include "nixl_communicator.hpp"
#include "registered_memory_resource.hpp"

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <mpi.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

void set_cuda_device()
{
  int mpi_rank;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

  int device_count;
  CUDA_RT_CALL(cudaGetDeviceCount(&device_count));
  std::cout << "Device count: " << device_count << std::endl;

  int current_device = mpi_rank % device_count;

  CUDA_RT_CALL(cudaSetDevice(current_device));
  std::cout << "Rank " << mpi_rank << " select " << current_device << "/" << device_count << " GPU"
            << std::endl;
}

size_t recommended_rmm_pool_size(size_t reserve_bytes)
{
  size_t free_memory, total_memory;
  CUDA_RT_CALL(cudaMemGetInfo(&free_memory, &total_memory));

  size_t pool_size =
    (reserve_bytes > 0 && reserve_bytes < free_memory) ? (free_memory - reserve_bytes)
                                                       : (free_memory / 284 * 256);

  int device = 0;
  cudaDeviceProp prop{};
  CUDA_RT_CALL(cudaGetDevice(&device));
  CUDA_RT_CALL(cudaGetDeviceProperties(&prop, device));
  // GB10 reports the full unified CPU/GPU RAM via cudaMemGetInfo. Leaving only
  // 500MB (the old discrete-GPU heuristic) OOMs the box.
  if (prop.integrated) { pool_size = std::min(pool_size, total_memory / 8); }
  pool_size = pool_size / 256 * 256;

  return pool_size;
}

void setup_memory_pool_and_communicator(
  Communicator *&communicator,
  registered_memory_resource *&registered_mr,
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *&pool_mr,
  std::string communicator_name,
  std::string registration_method,
  int64_t communicator_buffer_size)
{
  int mpi_size;
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  registered_mr = nullptr;

  const size_t pool_size = recommended_rmm_pool_size();

  if (communicator_name == "NCCL") {
    communicator = new NCCLCommunicator;
    communicator->initialize();
    pool_mr = new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
      rmm::mr::get_current_device_resource(), pool_size, pool_size);
    rmm::mr::set_current_device_resource(pool_mr);
  } else if (communicator_name == "NIXL") {
    auto *nixl_communicator = new NIXLCommunicator;
    communicator            = nixl_communicator;
    nixl_communicator->initialize();
    // Pinned host pages and the NIC's ibv_reg_mr of the same pool both count against
    // RLIMIT_MEMLOCK (~15 GiB on the zeno boxes, not raisable), so the pool must stay under half.
    size_t nixl_pool_size = pool_size;
    if (nixl_communicator->host_pool) {
      char const *gib  = std::getenv("NIXL_HOST_POOL_GIB");
      size_t const cap = (gib ? std::strtoull(gib, nullptr, 10) : size_t{6}) << 30;
      nixl_pool_size   = std::min(pool_size, cap);
    }
    pool_mr = new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
      nixl_communicator->memory_resource(), nixl_pool_size, nixl_pool_size);
    rmm::mr::set_current_device_resource(pool_mr);
    nixl_communicator->publish_registered_memory();
  } else if (communicator_name == "UCX") {
    if (registration_method == "buffer") {
      // For UCX with buffer communicator, a memory pool is first constructed so that the
      // communication buffers will be allocated in memory pool.
      pool_mr = new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
        rmm::mr::get_current_device_resource(), pool_size, pool_size);
      rmm::mr::set_current_device_resource(pool_mr);
      // *2 because buffers are needed for both sends and receives
      const int num_comm_buffers = 2 * mpi_size;
      communicator               = initialize_ucx_communicator(
        true, num_comm_buffers, communicator_buffer_size / num_comm_buffers - 100'000LL);
    } else if (registration_method == "preregistered") {
      // For UCX with preregistered memory pool, a communicator is first constructed so that
      // `registered_memory_resource` can use the communicator for buffer registrations.
      UCXCommunicator *ucx_communicator = initialize_ucx_communicator(false, 0, 0);
      communicator                      = ucx_communicator;
      registered_mr                     = new registered_memory_resource(ucx_communicator);
      pool_mr = new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
        *registered_mr, pool_size, pool_size);
      rmm::mr::set_current_device_resource(pool_mr);
    } else if (registration_method == "none") {
      communicator = initialize_ucx_communicator(false, 0, 0);
      pool_mr      = new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
        rmm::mr::get_current_device_resource(), pool_size, pool_size);
      rmm::mr::set_current_device_resource(pool_mr);
    } else {
      throw std::runtime_error("Unknown registration method");
    }
  } else {
    throw std::runtime_error("Unknown communicator name");
  }
}

void destroy_memory_pool_and_communicator(
  Communicator *communicator,
  registered_memory_resource *registered_mr,
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *pool_mr,
  std::string communicator_name,
  std::string registration_method)
{
  if (communicator_name == "UCX" && registration_method == "buffer") {
    // When finalizing buffer communicator, communication buffers need be deallocated, so
    // `finalize` needs to be called before the memory pool is deleted.
    communicator->finalize();
    delete pool_mr;
    delete registered_mr;
  } else {
    // For registered memory resouce, the memory pool needs to be deleted before finalizing
    // the communicator, so that all buffers can be deregistered through UCX.
    // For every other scenario, the order of deleting memory pool and finalizing the communicator
    // does not matter, and we just choose this path.
    delete pool_mr;
    delete registered_mr;
    communicator->finalize();
  }

  delete communicator;
}
